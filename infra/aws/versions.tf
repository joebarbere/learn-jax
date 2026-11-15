terraform {
  required_version = ">= 1.6"
  required_providers {
    aws    = { source = "hashicorp/aws", version = "~> 5.70" }
    random = { source = "hashicorp/random", version = "~> 3.6" }
  }
  # Local state (gitignored). This is a short-lived study stack: apply, run, destroy.
}

provider "aws" {
  region  = var.region
  profile = var.aws_profile
  default_tags {
    tags = {
      Project   = "learn-jax" # cost-allocation tag → the learn-jax-monthly budget in aws-cloud
      ManagedBy = "terraform"
      Module    = "25-aws"
    }
  }
}
