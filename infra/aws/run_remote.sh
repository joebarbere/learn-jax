#!/usr/bin/env bash
# Run Module 25's benchmarks on the instances `terraform apply` created, and bring the results home.
#
#   infra/aws/run_remote.sh            # what to run is chosen from the Terraform profile:
#                                      #   gpu1 / gpu1-l40s → bench_single     gpu4 → bench_single + bench_multi
#                                      #   multihost        → bench_multihost on both nodes    trn → bench_single
#
# How: the course code (learnjax/ + the modules the benchmarks import) is tarred and uploaded to the
# session's private S3 bucket; `aws ssm send-command` (no SSH, no open ports) tells each instance to fetch it,
# set up JAX for its accelerator, run the benchmark and upload runs/25-aws/*.json; this script waits, then
# downloads the JSON into runs/25-aws/ for the Module 25 lesson to compare with your RX 7600 XT.
#
# JAX on each stack (sources):
#   NVIDIA: "jax[cuda13]" — JAX installation docs, https://docs.jax.dev/en/latest/installation.html
#           (CUDA 13 wheels bundle the CUDA libraries; the DLAMI provides the NVIDIA driver), installed into
#           a venv with uv (https://docs.astral.sh/uv/): a pinned release from GitHub, checked against its
#           published SHA-256. Much faster than pip, which matters at $/hour.
#   Neuron: the Neuron JAX DLAMI's venv, `source /opt/aws_neuronx_venv_jax/bin/activate` — AWS Neuron docs,
#           https://awsdocs-neuron.readthedocs-hosted.com/en/latest/setup/jax/dlami.html (JAX NeuronX is beta)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
# AWS_PROFILE: yours from the environment, else the one the stack was applied with (terraform.tfvars).
if [[ -z "${AWS_PROFILE:-}" ]]; then
  p="$(terraform -chdir="$here" output -raw aws_profile 2>/dev/null || true)"
  [[ -n "$p" ]] && export AWS_PROFILE="$p"
fi
export AWS_REGION="${AWS_REGION:-us-east-1}" AWS_PAGER=""
JAX_VERSION="${JAX_VERSION:-0.11.1}"   # match the course container
UV_VERSION="${UV_VERSION:-0.13.6}"     # match containers/lock.sh
TIMEOUT_S="${TIMEOUT_S:-3000}"

tf() { terraform -chdir="$here" output -raw "$1"; }
tfj() { terraform -chdir="$here" output -json "$1"; }
profile="$(tf profile)"
bucket="$(tf results_bucket)"
mapfile -t ids < <(tfj instance_ids | python3 -c 'import json,sys; print("\n".join(json.load(sys.stdin)))')
mapfile -t ips < <(tfj private_ips | python3 -c 'import json,sys; print("\n".join(json.load(sys.stdin)))')
echo "profile=$profile instances=${ids[*]} bucket=$bucket"

# 1. ship the code the benchmarks need
tarball="$(mktemp --suffix=.tgz)"
tar -C "$root" -czf "$tarball" --exclude='__pycache__' --exclude='__marimo__' \
  learnjax modules/16-gpt modules/18-inference modules/24-parallelism modules/25-aws
aws s3 cp --only-show-errors "$tarball" "s3://$bucket/code/learn-jax.tgz"
rm -f "$tarball"

# 2. the shell each instance runs (as root via SSM); $1 = the benchmark command line
remote_script() {
  local bench="$1"
  cat <<EOF
set -euo pipefail
source /opt/learn-jax/node.env
mkdir -p /opt/learn-jax/course && cd /opt/learn-jax/course
aws s3 cp --only-show-errors s3://\$LEARNJAX_BUCKET/code/learn-jax.tgz /tmp/learn-jax.tgz
tar -xzf /tmp/learn-jax.tgz
export HF_HOME=/opt/learn-jax/hf LEARNJAX_CACHE=/opt/learn-jax/cache PYTHONPATH=/opt/learn-jax/course PYTHONUNBUFFERED=1
if [ "\$LEARNJAX_STACK" = neuron ]; then
  source /opt/aws_neuronx_venv_jax/bin/activate
  pip install -q optax safetensors tokenizers huggingface_hub
else
  if [ ! -x /opt/learn-jax/bin/uv ]; then
    rel=https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-x86_64-unknown-linux-gnu.tar.gz
    curl -fsSL -o /tmp/uv.tgz "\$rel" && curl -fsSL -o /tmp/uv.tgz.sha256 "\$rel.sha256"
    echo "\$(cut -d' ' -f1 /tmp/uv.tgz.sha256)  /tmp/uv.tgz" | sha256sum -c --quiet
    mkdir -p /opt/learn-jax/bin && tar -xzf /tmp/uv.tgz -C /opt/learn-jax/bin --strip-components=1
  fi
  export UV_CACHE_DIR=/opt/learn-jax/uv-cache
  [ -d /opt/learn-jax/venv ] || /opt/learn-jax/bin/uv venv --quiet --python python3 /opt/learn-jax/venv
  source /opt/learn-jax/venv/bin/activate
  /opt/learn-jax/bin/uv pip install --quiet "jax[cuda13]==$JAX_VERSION" optax==0.2.9 safetensors tokenizers huggingface_hub
fi
python -c "import jax; print('JAX', jax.__version__, jax.devices())"
cd modules/25-aws && $bench
aws s3 cp --only-show-errors --recursive /opt/learn-jax/course/runs/25-aws s3://\$LEARNJAX_BUCKET/results/
EOF
}

send() {  # send <instance-id> <bench command> → prints command id
  local params
  params="$(python3 -c 'import json,sys; print(json.dumps({"commands": sys.stdin.read().splitlines(), "executionTimeout": [sys.argv[1]]}))' "$TIMEOUT_S" <<<"$(remote_script "$2")")"
  aws ssm send-command --instance-ids "$1" --document-name AWS-RunShellScript --comment "learn-jax $profile" \
    --timeout-seconds 600 --parameters "$params" --query Command.CommandId --output text
}

wait_for() {  # wait_for <command-id> <instance-id>
  local status
  while true; do
    status="$(aws ssm get-command-invocation --command-id "$1" --instance-id "$2" --query Status --output text 2>/dev/null || echo Pending)"
    case "$status" in
      Success) echo "  $2: Success"; return 0 ;;
      Failed|Cancelled|TimedOut|Cancelling)
        echo "  $2: $status — last output:"
        aws ssm get-command-invocation --command-id "$1" --instance-id "$2" \
          --query '[StandardOutputContent,StandardErrorContent]' --output text | tail -40
        return 1 ;;
      *) sleep 15 ;;
    esac
  done
}

# 3. wait until SSM sees the instances (first boot takes a few minutes)
for id in "${ids[@]}"; do
  echo "waiting for SSM on $id ..."
  until [ "$(aws ssm describe-instance-information --filters "Key=InstanceIds,Values=$id" \
             --query 'length(InstanceInformationList)' --output text)" = 1 ]; do sleep 10; done
done

# 4. run
declare -a cmds
case "$profile" in
  gpu1|gpu1-l40s|trn) cmds=("$(send "${ids[0]}" "python bench_single.py")") ;;
  gpu4)              cmds=("$(send "${ids[0]}" "python bench_single.py && python bench_multi.py")") ;;
  multihost)
    coord="${ips[0]}:12355"
    for i in "${!ids[@]}"; do
      cmds+=("$(send "${ids[$i]}" "python bench_single.py --no-decode && python bench_multihost.py --coordinator $coord --num-processes ${#ids[@]} --process-id $i")")
    done ;;
esac
rc=0
for i in "${!cmds[@]}"; do wait_for "${cmds[$i]}" "${ids[$i]}" || rc=1; done

# 5. bring results home
mkdir -p "$root/runs/25-aws"
aws s3 cp --only-show-errors --recursive "s3://$bucket/results/" "$root/runs/25-aws/" || true
ls -l "$root/runs/25-aws/"
echo "Done. Now: terraform -chdir=infra/aws destroy   (instances also self-terminate after max_minutes)"
exit $rc
