output "profile" {
  value = var.profile
}

output "instance_ids" {
  value = aws_instance.node[*].id
}

output "instance_types" {
  value = aws_instance.node[*].instance_type
}

output "private_ips" {
  description = "Rank 0's private IP is the jax.distributed coordinator for the multihost profile."
  value       = aws_instance.node[*].private_ip
}

output "results_bucket" {
  value = aws_s3_bucket.results.id
}

output "ssm_sessions" {
  description = "Interactive shells (no SSH, no open ports)."
  value       = [for i in aws_instance.node : "aws ssm start-session --target ${i.id} --region ${var.region} --profile ${var.aws_profile}"]
}

output "auto_terminates_after_minutes" {
  value = var.max_minutes
}

output "spot" {
  value = var.use_spot
}
