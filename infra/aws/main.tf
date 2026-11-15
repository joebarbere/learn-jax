locals {
  profiles = {
    gpu1        = { types = ["g6.xlarge"], ami = "nvidia" }
    "gpu1-l40s" = { types = ["g6e.xlarge"], ami = "nvidia" }
    multihost   = { types = ["g6.xlarge", "g6.xlarge"], ami = "nvidia" }
    gpu4        = { types = ["g6.12xlarge"], ami = "nvidia" }
    trn         = { types = ["trn1.2xlarge"], ami = "neuron" }
  }
  p = local.profiles[var.profile]

  # The seatbelt's EC2 allowlist (empty list, so no plan-time check, when seatbelt_path isn't set).
  seatbelt = try(jsondecode(file(var.seatbelt_path)), null)
  allowlist = local.seatbelt == null ? [] : flatten([
    for s in local.seatbelt.Statement : s.Condition.StringNotEquals["ec2:InstanceType"]
    if s.Sid == "Ec2InstanceTypeAllowlist"
  ])
}

# ---------------------------------------------------------------- AMIs (AWS-published, via SSM parameters)
# NVIDIA: Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04) — driver only; JAX comes from pip.
data "aws_ssm_parameter" "nvidia_ami" {
  name = "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id"
}
# Neuron: Deep Learning AMI Neuron JAX (Ubuntu 24.04) — JAX + jax-neuronx preinstalled in
# /opt/aws_neuronx_venv_jax (AWS Neuron docs, "Set up JAX on a Neuron DLAMI").
data "aws_ssm_parameter" "neuron_ami" {
  name = "/aws/service/neuron/dlami/jax-0.10/ubuntu-24.04/latest/image_id"
}

# ---------------------------------------------------------------- network: public subnet, no NAT
resource "aws_vpc" "this" {
  cidr_block           = "10.42.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true
  tags                 = { Name = "learn-jax" }
}

resource "aws_internet_gateway" "this" {
  vpc_id = aws_vpc.this.id
  tags   = { Name = "learn-jax" }
}

resource "aws_subnet" "public" {
  vpc_id                  = aws_vpc.this.id
  cidr_block              = "10.42.1.0/24"
  availability_zone       = var.availability_zone
  map_public_ip_on_launch = true # outbound internet (pip, Hugging Face, SSM) without a NAT gateway
  tags                    = { Name = "learn-jax-public" }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.this.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.this.id
  }
  tags = { Name = "learn-jax-public" }
}

resource "aws_route_table_association" "public" {
  subnet_id      = aws_subnet.public.id
  route_table_id = aws_route_table.public.id
}

# No inbound from the internet at all (no SSH): access is SSM Session Manager. Instances may talk to each
# other on any port (jax.distributed's coordinator and the collectives between hosts).
resource "aws_security_group" "node" {
  name        = "learn-jax-node"
  description = "learn-jax: intra-group only; egress to the internet"
  vpc_id      = aws_vpc.this.id
  tags        = { Name = "learn-jax-node" }
}

resource "aws_vpc_security_group_ingress_rule" "intra" {
  security_group_id            = aws_security_group.node.id
  referenced_security_group_id = aws_security_group.node.id
  ip_protocol                  = "-1"
  description                  = "multi-host JAX between learn-jax nodes"
}

resource "aws_vpc_security_group_egress_rule" "all" {
  security_group_id = aws_security_group.node.id
  cidr_ipv4         = "0.0.0.0/0"
  ip_protocol       = "-1"
  description       = "pip, Hugging Face, S3, SSM"
}

# ---------------------------------------------------------------- results bucket
resource "random_id" "bucket" {
  byte_length = 4
}

resource "aws_s3_bucket" "results" {
  bucket        = "learn-jax-results-${random_id.bucket.hex}"
  force_destroy = true # a scratch bucket: `terraform destroy` empties it
}

resource "aws_s3_bucket_public_access_block" "results" {
  bucket                  = aws_s3_bucket.results.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "results" {
  bucket = aws_s3_bucket.results.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "AES256" }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "results" {
  bucket = aws_s3_bucket.results.id
  rule {
    id     = "expire-30d"
    status = "Enabled"
    filter {}
    expiration { days = 30 }
  }
}

# ---------------------------------------------------------------- instance role: SSM + this bucket only
data "aws_iam_policy_document" "assume_ec2" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "node" {
  name               = "learn-jax-node"
  assume_role_policy = data.aws_iam_policy_document.assume_ec2.json
}

resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.node.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

data "aws_iam_policy_document" "bucket" {
  statement {
    actions   = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
    resources = [aws_s3_bucket.results.arn, "${aws_s3_bucket.results.arn}/*"]
  }
}

resource "aws_iam_role_policy" "bucket" {
  name   = "learn-jax-results-bucket"
  role   = aws_iam_role.node.id
  policy = data.aws_iam_policy_document.bucket.json
}

resource "aws_iam_instance_profile" "node" {
  name = "learn-jax-node"
  role = aws_iam_role.node.name
}

# ---------------------------------------------------------------- the instances
resource "aws_instance" "node" {
  count                  = length(local.p.types)
  instance_type          = local.p.types[count.index]
  ami                    = local.p.ami == "neuron" ? data.aws_ssm_parameter.neuron_ami.value : data.aws_ssm_parameter.nvidia_ami.value
  subnet_id              = aws_subnet.public.id
  vpc_security_group_ids = [aws_security_group.node.id]
  iam_instance_profile   = aws_iam_instance_profile.node.name

  # Hard auto-stop, part 1: a shutdown from inside the instance *terminates* it (no stopped instance or
  # volume left behind). Part 2 is the `shutdown -h +N` in user_data below.
  instance_initiated_shutdown_behavior = "terminate"

  user_data = templatefile("${path.module}/user_data.sh.tftpl", {
    max_minutes = var.max_minutes
    rank        = count.index
    n_nodes     = length(local.p.types)
    bucket      = aws_s3_bucket.results.id
    stack       = local.p.ami
  })

  metadata_options {
    http_tokens   = "required" # IMDSv2 only
    http_endpoint = "enabled"
  }

  root_block_device {
    volume_type           = "gp3"
    volume_size           = var.root_volume_gb
    encrypted             = true
    delete_on_termination = true
  }

  dynamic "instance_market_options" {
    for_each = var.use_spot ? [1] : []
    content {
      market_type = "spot"
      spot_options {
        spot_instance_type             = "one-time"
        instance_interruption_behavior = "terminate"
      }
    }
  }

  tags = { Name = "learn-jax-${var.profile}-${count.index}", Rank = tostring(count.index) }

  lifecycle {
    precondition {
      condition     = length(local.allowlist) == 0 || contains(local.allowlist, local.p.types[count.index])
      error_message = "Instance type ${local.p.types[count.index]} is not in the seatbelt allowlist (${var.seatbelt_path}). Add it to the seatbelt policy and apply that first."
    }
  }
}
