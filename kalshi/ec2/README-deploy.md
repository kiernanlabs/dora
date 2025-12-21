# Dora Bot EC2 Deployment

Use `ec2/deploy-to-ec2.sh` to deploy from your laptop; keep `ec2/setup-ec2.sh` as a manual fallback you can run directly on the instance.

## Prerequisites

- Laptop: repo checked out, `ssh`, `scp`, `rsync`, and your EC2 SSH key (.pem).
- EC2: Amazon Linux 2023 or Ubuntu 22.04 reachable over SSH
- **IAM Role**: EC2 instance must have an IAM role with:
  - DynamoDB access (for bot state/config)
  - `CloudWatchAgentServerPolicy` managed policy (for logs/metrics)
- Default remote user is `ec2-user`; if you use a different user (e.g., Ubuntu images), edit `REMOTE_USER` in `ec2/deploy-to-ec2.sh` before running.

## First-Time Bootstrap (recommended)

Run from the repo root:

```bash
chmod +x kalshi/ec2/deploy-to-ec2.sh
./kalshi/ec2/deploy-to-ec2.sh <EC2_IP> <PATH_TO_PEM> --setup
```

What the script does:
- Updates the instance and installs Python 3.11, pip, git, and rsync.
- Syncs your repo to `/home/ec2-user/dora/kalshi` (excludes secrets, venvs, and git).
- Creates `/home/ec2-user/dora/venv` and installs requirements from `kalshi/dora_bot/requirements.txt`.
- Installs/enables the systemd unit from `kalshi/ec2/dora-bot.service`.
- **Sets up CloudWatch agent** (installs, configures, sets permissions, starts agent).

## Provide Credentials (required before the service can run)

1) Base64-encode your private key locally (single line, no newlines):
```bash
base64 -w0 path/to/your-private-key.pem > key.b64
```
Copy the contents of `key.b64` for the next step.

2) Create an environment file on the instance and tell systemd to load it:
```bash
ssh -i <PATH_TO_PEM> ec2-user@<EC2_IP>

# On the instance:
cat <<'EOF' | sudo tee /etc/dora-bot-demo.env >/dev/null
KALSHI_KEY_ID=your-key-id
KALSHI_PRIVATE_KEY=<paste-the-base64-private-key-here>
USE_DEMO=true                        # set to false for prod
AWS_REGION=us-east-1
EOF

sudo mkdir -p /etc/systemd/system/dora-bot.service.d
printf "[Service]\nEnvironmentFile=/etc/dora-bot-demo.env\n" | sudo tee /etc/systemd/system/dora-bot.service.d/env.conf >/dev/null
sudo systemctl daemon-reload
sudo systemctl restart dora-bot
```

### Demo and Prod Keys (recommended)

Keep separate env files and switch the systemd drop-in when you change environments:

```bash
# Demo
cat <<'EOF' | sudo tee /etc/dora-bot-demo.env >/dev/null
KALSHI_KEY_ID=your-demo-key-id
KALSHI_PRIVATE_KEY=<paste-demo-base64-key-here>
USE_DEMO=true
AWS_REGION=us-east-1
EOF

# Prod
cat <<'EOF' | sudo tee /etc/dora-bot-prod.env >/dev/null
KALSHI_KEY_ID=your-prod-key-id
KALSHI_PRIVATE_KEY=<paste-prod-base64-key-here>
USE_DEMO=false
AWS_REGION=us-east-1
EOF

# Point systemd at the env you want
printf "[Service]\nEnvironmentFile=/etc/dora-bot-demo.env\n" | sudo tee /etc/systemd/system/dora-bot.service.d/env.conf >/dev/null
sudo systemctl daemon-reload
sudo systemctl restart dora-bot
```

To switch to prod later, update the drop-in to `/etc/dora-bot-prod.env` and restart.

## Update / Redeploy

For code or config updates (after the first setup), rerun without `--setup`:
```bash
./kalshi/ec2/deploy-to-ec2.sh <EC2_IP> <PATH_TO_PEM>
```
The script resyncs code, reinstalls Python deps, refreshes the systemd unit, and restarts the service.

## Manage the Service

- Status: `ssh -i <PATH_TO_PEM> ec2-user@<EC2_IP> "sudo systemctl status dora-bot --no-pager"`
- Logs (follow): `ssh -i <PATH_TO_PEM> ec2-user@<EC2_IP> "tail -f /home/ec2-user/dora/kalshi/logs/dora-bot.log"`
- Restart: `ssh -i <PATH_TO_PEM> ec2-user@<EC2_IP> "sudo systemctl restart dora-bot"`

## CloudWatch Logs

The deploy script **automatically** sets up the CloudWatch agent to ship logs to CloudWatch. Logs will appear under:
- Demo environment: `/dora-bot/demo/app`
- Prod environment: `/dora-bot/prod/app`

**Important**: Your EC2 instance **must have an IAM role** with the `CloudWatchAgentServerPolicy` managed policy attached. Without this, the agent will fail to send logs.

The script handles:
- Installing the CloudWatch agent
- Copying and configuring `cloudwatch-agent-config.json`
- Setting directory permissions (755) so the `cwagent` user can read log files
- Starting the agent with the correct config

To switch between demo and prod environments, edit line 201 in `deploy-to-ec2.sh`:
```bash
sudo sed -i 's/{env}/demo/g' ...  # Change 'demo' to 'prod' for production
```

## Manual Fallback (if you cannot run the deploy script)

Copy `kalshi/ec2/setup-ec2.sh` to the instance, run it there to install Python/venv/deps, then:
1) Sync the repo into `/home/ec2-user/dora/kalshi`.
2) Install the systemd unit from `kalshi/ec2/dora-bot.service`.
3) Add `/etc/dora-bot.env` and the `EnvironmentFile` drop-in as shown above.
4) Reload systemd and restart `dora-bot`.
