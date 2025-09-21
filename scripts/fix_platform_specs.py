#!/usr/bin/env python3
"""
Fix missing platform specifications in Docker Compose files.
Adds platform: linux/amd64 for THANOS services and platform: linux/arm64 for ORACLE1 services.
"""

import yaml
import re
from pathlib import Path

def add_platform_specs(compose_file, default_platform):
    """Add platform specifications to services missing them."""

    with open(compose_file, 'r') as f:
        content = f.read()
        data = yaml.safe_load(content)

    if not data or 'services' not in data:
        print(f"No services found in {compose_file}")
        return

    services_fixed = 0
    lines = content.split('\n')
    new_lines = []

    in_service = False
    current_service = None
    has_platform = False
    indent = "  "  # Default indent

    for i, line in enumerate(lines):
        # Check if we're starting a new service definition
        if line and not line.startswith(' ') and not line.startswith('\t'):
            # Top level key
            if line.startswith('services:'):
                in_service = True
        elif in_service and line and not line.startswith('    '):
            # Service name (2 spaces indent)
            match = re.match(r'^(\s{2})([a-zA-Z0-9_-]+):$', line)
            if match:
                # Check if previous service needed platform
                if current_service and not has_platform:
                    # Insert platform spec after image line
                    for j in range(len(new_lines) - 1, -1, -1):
                        if 'image:' in new_lines[j]:
                            # Insert platform after image
                            new_lines.insert(j + 1, f"{indent}  platform: {default_platform}")
                            services_fixed += 1
                            break

                current_service = match.group(2)
                has_platform = False
                indent = match.group(1)

        # Check if current line has platform
        if 'platform:' in line and current_service:
            has_platform = True

        new_lines.append(line)

    # Handle last service
    if current_service and not has_platform:
        for j in range(len(new_lines) - 1, -1, -1):
            if f'{current_service}:' in new_lines[j]:
                # Find the image line for this service
                for k in range(j + 1, min(j + 20, len(new_lines))):
                    if 'image:' in new_lines[k]:
                        new_lines.insert(k + 1, f"{indent}  platform: {default_platform}")
                        services_fixed += 1
                        break
                break

    if services_fixed > 0:
        # Write back the fixed content
        with open(compose_file, 'w') as f:
            f.write('\n'.join(new_lines))
        print(f"Fixed {services_fixed} services in {compose_file}")
    else:
        print(f"No fixes needed in {compose_file}")

    return services_fixed

def main():
    """Fix platform specifications in all Docker Compose files."""

    # Fix THANOS compose files (AMD64)
    thanos_files = [
        'docker-compose-thanos-unified.yml',
        'docker-compose-phase7.yml',
        'docker-compose-phase8.yml',
        'docker-compose-phase9.yml'
    ]

    # Fix ORACLE1 compose files (ARM64)
    oracle_files = [
        'docker-compose-oracle1-unified.yml'
    ]

    total_fixed = 0

    print("Fixing THANOS services (linux/amd64)...")
    for file in thanos_files:
        if Path(file).exists():
            total_fixed += add_platform_specs(file, 'linux/amd64')

    print("\nFixing ORACLE1 services (linux/arm64)...")
    for file in oracle_files:
        if Path(file).exists():
            total_fixed += add_platform_specs(file, 'linux/arm64')

    print(f"\nTotal services fixed: {total_fixed}")

if __name__ == "__main__":
    main()