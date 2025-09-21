#!/usr/bin/env python3
"""
Fix Vault configuration in docker-compose files
- Remove Vault service from THANOS and ORACLE1 compose files
- Update VAULT_ADDR to use Tailscale IP
"""

import yaml
import sys
from pathlib import Path

def fix_compose_file(filepath, remove_vault=True, vault_ip="100.122.12.35"):
    """Fix a docker-compose file to use proper Vault configuration"""

    print(f"Processing {filepath}...")

    # Read the compose file
    with open(filepath, 'r') as f:
        compose = yaml.safe_load(f)

    changes_made = False

    # Remove vault service if it exists
    if remove_vault and 'services' in compose and 'vault' in compose['services']:
        print(f"  - Removing vault service from {filepath}")
        del compose['services']['vault']
        changes_made = True

    # Update VAULT_ADDR in all services
    if 'services' in compose:
        for service_name, service_config in compose['services'].items():
            if 'environment' in service_config:
                env = service_config['environment']

                # Handle both list and dict formats
                if isinstance(env, list):
                    new_env = []
                    for var in env:
                        if var.startswith('VAULT_ADDR='):
                            old_value = var.split('=', 1)[1]
                            if 'vault:8200' in old_value or 'localhost:8200' in old_value:
                                new_value = f'VAULT_ADDR=http://{vault_ip}:8200'
                                print(f"  - Updating {service_name}: {var} -> {new_value}")
                                new_env.append(new_value)
                                changes_made = True
                            else:
                                new_env.append(var)
                        else:
                            new_env.append(var)
                    service_config['environment'] = new_env

                elif isinstance(env, dict):
                    if 'VAULT_ADDR' in env:
                        old_value = env['VAULT_ADDR']
                        if 'vault:8200' in old_value or 'localhost:8200' in old_value:
                            new_value = f'http://{vault_ip}:8200'
                            print(f"  - Updating {service_name}: VAULT_ADDR={old_value} -> VAULT_ADDR={new_value}")
                            env['VAULT_ADDR'] = new_value
                            changes_made = True

    # Remove vault from depends_on if it exists
    if 'services' in compose:
        for service_name, service_config in compose['services'].items():
            if 'depends_on' in service_config:
                deps = service_config['depends_on']
                if isinstance(deps, list) and 'vault' in deps:
                    print(f"  - Removing vault dependency from {service_name}")
                    deps.remove('vault')
                    changes_made = True
                elif isinstance(deps, dict) and 'vault' in deps:
                    print(f"  - Removing vault dependency from {service_name}")
                    del deps['vault']
                    changes_made = True

    if changes_made:
        # Backup original file
        backup_path = f"{filepath}.backup"
        print(f"  - Creating backup: {backup_path}")
        import shutil
        shutil.copy2(filepath, backup_path)

        # Write the fixed compose file
        with open(filepath, 'w') as f:
            yaml.dump(compose, f, default_flow_style=False, sort_keys=False)
        print(f"  ✅ Fixed {filepath}")
    else:
        print(f"  ℹ️  No changes needed for {filepath}")

    return changes_made

def main():
    print("BEV Vault Configuration Fixer")
    print("=" * 50)

    # Fix THANOS compose file
    thanos_file = Path("docker-compose-thanos-unified.yml")
    if thanos_file.exists():
        fix_compose_file(thanos_file, remove_vault=True, vault_ip="100.122.12.35")
    else:
        print(f"❌ {thanos_file} not found!")

    print()

    # Fix ORACLE1 compose file
    oracle_file = Path("docker-compose-oracle1-unified.yml")
    if oracle_file.exists():
        fix_compose_file(oracle_file, remove_vault=True, vault_ip="100.122.12.35")
    else:
        print(f"❌ {oracle_file} not found!")

    print()
    print("✅ Vault configuration fix complete!")
    print("\nNext steps:")
    print("1. Deploy Vault on STARLORD only: docker-compose -f docker-compose-starlord-vault.yml up -d")
    print("2. Deploy THANOS services: ssh thanos 'cd /opt/bev && docker-compose -f docker-compose-thanos-unified.yml up -d'")
    print("3. Deploy ORACLE1 services: ssh oracle1 'cd /opt/bev && docker-compose -f docker-compose-oracle1-unified.yml up -d'")

if __name__ == "__main__":
    main()