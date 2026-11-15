variable "region" {
  description = "Must be us-east-1: the aws-cloud seatbelt denies every other region."
  type        = string
  default     = "us-east-1"
}

variable "aws_profile" {
  description = "AWS CLI profile (IAM Identity Center)."
  type        = string
  default     = "joebarbere-admin"
}

variable "profile" {
  description = <<-EOT
    Which session to run (see README.md for cost):
      gpu1      1 × g6.xlarge   (NVIDIA L4 24 GB)     — portability: same JAX code on CUDA
      gpu1-l40s 1 × g6e.xlarge  (NVIDIA L40S 48 GB)   — portability, bigger GPU
      multihost 2 × g6.xlarge                          — jax.distributed across two machines
      gpu4      1 × g6.12xlarge (4 × L4)               — real multi-GPU DP / FSDP / TP
      trn       1 × trn1.2xlarge (Trainium, 2 cores)   — JAX on Neuron (jax-neuronx, beta)
  EOT
  type        = string
  default     = "gpu1"
  validation {
    condition     = contains(["gpu1", "gpu1-l40s", "multihost", "gpu4", "trn"], var.profile)
    error_message = "profile must be one of gpu1, gpu1-l40s, multihost, gpu4, trn."
  }
}

variable "use_spot" {
  description = <<-EOT
    Launch as one-time spot via RunInstances market options (the seatbelt denies RequestSpotInstances and
    fleets, not this). Default false: on 2026-11-14 the account's spot quotas for G and Trn instances were 0
    (increases requested). Set true once they are approved — spot is typically far cheaper, but can be
    interrupted (the instance then terminates; rerun the session).
  EOT
  type        = bool
  default     = false
}

variable "max_minutes" {
  description = <<-EOT
    Hard auto-stop. User data runs `shutdown -h +max_minutes` at boot, and the instance's shutdown behavior is
    "terminate", so the instance (and its root volume) is gone after this many minutes even if you forget to
    `terraform destroy`. Budgets only alert; this is what actually stops spend on compute.
  EOT
  type        = number
  default     = 120
  validation {
    condition     = var.max_minutes >= 10 && var.max_minutes <= 480
    error_message = "max_minutes must be between 10 and 480."
  }
}

variable "availability_zone" {
  description = "us-east-1c offers every type here (trn1.2xlarge is only in 1c and 1f; checked 2026-11-14)."
  type        = string
  default     = "us-east-1c"
}

variable "root_volume_gb" {
  description = "Root EBS size. The DLAMIs need ~75 GB+; models and pip caches need room."
  type        = number
  default     = 120
}

variable "seatbelt_path" {
  description = "aws-cloud's seatbelt policy, read to fail the plan early if an instance type isn't allowlisted."
  type        = string
  default     = "../../../aws-cloud/terraform/seatbelt.json"
}
