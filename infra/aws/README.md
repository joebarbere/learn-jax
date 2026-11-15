# infra/aws — paid sessions for Module 25

A short-lived study stack: **plan → you approve → apply → run → collect → destroy.** Nothing here runs by
itself, and nothing should be applied without the owner reading the plan first (see `CLAUDE.md`).

It creates, in us-east-1 (the only region the `aws-cloud` seatbelt allows):

- a VPC with one public subnet and an internet gateway. There's **no NAT gateway** (the seatbelt denies it)
  and **no inbound access from the internet**. You reach instances through SSM Session Manager. The security
  group only lets the session's own instances talk to each other (for multi-host JAX).
- 1–2 instances from an AWS Deep Learning AMI. The AMI is looked up through AWS's public SSM parameters,
  so it's always the latest:
  - NVIDIA: *Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)*, which has the driver. JAX comes
    from pip (`jax[cuda13]`, per the [JAX installation docs](https://docs.jax.dev/en/latest/installation.html)).
  - Trainium: *Deep Learning AMI Neuron JAX (Ubuntu 24.04)*, with JAX + `jax-neuronx` preinstalled in
    `/opt/aws_neuronx_venv_jax` (AWS Neuron docs:
    [JAX on a Neuron DLAMI](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/setup/jax/dlami.html)).
    JAX NeuronX is beta.
- IMDSv2-only instance metadata and an encrypted gp3 root volume. The instance role allows SSM plus this
  session's bucket and nothing else.
- a private, encrypted S3 bucket for code and results. It expires objects after 30 days and is emptied on destroy.
- tags `Project=learn-jax` and `ManagedBy=terraform` on everything, which feeds the `learn-jax-monthly` budget.

## Spending guards

1. **Hard auto-stop.** Each instance runs `shutdown -h +max_minutes` at boot (default 120). Its shutdown
   behavior is *terminate*, so after that time the instance and its disk are gone, even if you forget to
   destroy. Use `-var max_minutes=60` for a tighter session.
2. **Plan-time allowlist check.** The plan fails if a profile's instance type isn't in
   `../../../aws-cloud/terraform/seatbelt.json`, so you find out before anything is created.
3. **The seatbelt itself.** AWS refuses any instance type outside the allowlist, any other region, NAT,
   fleets, and capacity reservations.
4. **Budgets.** The `learn-jax-monthly` ($25) and account ($50) budgets email you. They do *not* stop anything.

## Profiles and cost per session

On-demand us-east-1 prices from instances.vantage.sh, 2026-11-14. A session is roughly 45–75 minutes:
boot, installing JAX with uv, first compiles, the benchmarks, then destroy. EBS adds about $0.02/h.

| `profile` | Instances | Teaches | $/hour | Typical session |
|---|---|---|---|---|
| `gpu1` | 1 × g6.xlarge (L4 24 GB) | same JAX code on CUDA; $/token vs your RX 7600 XT | 0.81 | **~$1** |
| `gpu1-l40s` | 1 × g6e.xlarge (L40S 48 GB) | a bigger, faster GPU | 1.86 | **~$2** |
| `multihost` | 2 × g6.xlarge | `jax.distributed` across machines; network vs in-box bandwidth | 1.61 | **~$2** |
| `gpu4` | 1 × g6.12xlarge (4 × L4) | real DP / FSDP / TP; collective bandwidth; comm/compute overlap | 4.60 | **~$5** |
| `trn` | 1 × trn1.2xlarge (Trainium, 2 NeuronCores) | JAX on a third accelerator family (beta) | 1.34 | **~$2** |

All five sessions together come to roughly **$12**.

**Needs, as of 2026-11-14:**
- `gpu4` and `trn` need the pending `aws-cloud` seatbelt change (branch `learn-jax-gpu`) to be applied.
  Until then the plan fails on the allowlist check.
- `trn` also needs the *Running On-Demand Trn instances* quota (requested: 8 vCPUs).
- Spot (`-var use_spot=true`) needs the G and Trn spot quotas (requested: 48 and 8). The default is
  on-demand until those are approved.

## A session

```bash
cd infra/aws
aws sso login --profile joebarbere-admin           # if the token expired
terraform init                                     # once
terraform plan -var profile=gpu1 -out=gpu1.tfplan  # read it: instance types, count, tags, no NAT
terraform apply gpu1.tfplan                        # ← the step that costs money (owner only)
./run_remote.sh                                    # ~20–40 min; results land in runs/25-aws/
terraform destroy -var profile=gpu1                # always; auto-stop is the safety net, not the plan
```

To look around on an instance: `terraform output ssm_sessions` prints `aws ssm start-session …` commands.
These need the [Session Manager plugin](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html)
from AWS. `run_remote.sh` itself only uses `send-command`, which doesn't.

Then open Module 25 (`make lesson N=25`). It finds each `runs/25-aws/<instance-type>.json` and compares it
with your card.

## Troubleshooting

- **`InsufficientInstanceCapacity`**: that AZ has no capacity right now. Try `-var availability_zone=us-east-1f`
  (trn1 is only in 1c and 1f) or wait.
- **`VcpuLimitExceeded` / `MaxSpotInstanceCountExceeded`**: a quota is still pending. Check with
  `aws service-quotas list-requested-service-quota-change-history --service-code ec2`.
- **`UnauthorizedOperation` on RunInstances**: the seatbelt refused the instance type. Apply the
  `aws-cloud` change first.
- **Spot interruption**: the instance terminates. Rerun the session; partial results already uploaded are kept.
