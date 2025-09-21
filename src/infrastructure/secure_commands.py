#!/usr/bin/env python3
"""
Secure command execution wrapper
Replaces unsafe # SECURITY: Replace with subprocess.run() - os.system() calls throughout BEV OSINT Framework
"""

import subprocess
import shlex
import os
import re
import logging
from typing import Optional, Tuple, List
from pathlib import Path

logger = logging.getLogger(__name__)

class SecureCommandExecutor:
    """
    Secure command execution with validation and logging
    NEVER use # SECURITY: Replace with subprocess.run() - os.system() - use this instead
    """

    ALLOWED_COMMANDS = {
        'sync', 'ls', 'ps', 'df', 'free', 'uptime', 'whoami',
        'docker', 'docker-compose', 'systemctl', 'journalctl',
        'redis-cli', 'psql', 'neo4j-admin', 'curl', 'wget'
    }

    DANGEROUS_PATTERNS = [
        r'[;&|`$()]',  # Command injection characters
        r'rm\s+-rf',   # Dangerous deletions
        r'sudo\s+',    # Privilege escalation
        r'su\s+',      # User switching
        r'chmod\s+777', # Unsafe permissions
        r'>/dev/null', # Output redirection that might hide errors
    ]

    def __init__(self, max_timeout: int = 30):
        self.max_timeout = max_timeout

    def is_command_safe(self, command: str) -> Tuple[bool, str]:
        """Validate command safety"""

        # Check for dangerous patterns
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, command):
                return False, f"Command contains dangerous pattern: {pattern}"

        # Extract base command
        args = shlex.split(command)
        if not args:
            return False, "Empty command"

        base_cmd = args[0]

        # Check if command is in allowed list
        if base_cmd not in self.ALLOWED_COMMANDS:
            return False, f"Command '{base_cmd}' not in allowed list"

        return True, "Command is safe"

    def execute(self, command: str, shell: bool = False, check: bool = True) -> Tuple[int, str, str]:
        """
        Secure command execution with validation

        Args:
            command: Command to execute
            shell: Whether to use shell execution (discouraged)
            check: Whether to raise exception on non-zero return

        Returns:
            Tuple of (return_code, stdout, stderr)
        """

        # Log the command execution attempt
        logger.info(f"Executing command: {command}")

        # Validate command safety
        is_safe, reason = self.is_command_safe(command)
        if not is_safe:
            logger.error(f"Unsafe command blocked: {reason}")
            raise SecurityError(f"Command blocked for security: {reason}")

        if shell:
            logger.warning(f"Shell execution requested for: {command}")

        try:
            if shell:
                # Only for explicitly approved commands
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=self.max_timeout,
                    check=check
                )
            else:
                # Safer: parse arguments
                args = shlex.split(command)
                result = subprocess.run(
                    args,
                    capture_output=True,
                    text=True,
                    timeout=self.max_timeout,
                    check=check
                )

            logger.info(f"Command completed with return code: {result.returncode}")
            return result.returncode, result.stdout, result.stderr

        except subprocess.TimeoutExpired:
            error_msg = f"Command timed out after {self.max_timeout}s: {command}"
            logger.error(error_msg)
            return -1, "", error_msg
        except subprocess.CalledProcessError as e:
            error_msg = f"Command failed with code {e.returncode}: {command}"
            logger.error(error_msg)
            return e.returncode, e.stdout or "", e.stderr or ""
        except Exception as e:
            error_msg = f"Command execution failed: {e}"
            logger.error(error_msg)
            return -1, "", str(e)

class SecurityError(Exception):
    """Raised when a security violation is detected"""
    pass

# Global secure executor instance
secure_executor = SecureCommandExecutor()

def secure_execute(command: str, shell: bool = False, check: bool = True) -> Tuple[int, str, str]:
    """
    Convenience function for secure command execution

    Usage:
        return_code, stdout, stderr = secure_execute('ls -la')

    Replaces:
        # SECURITY: Replace with subprocess.run() - os.system('ls -la')  # NEVER USE THIS
    """
    return secure_executor.execute(command, shell=shell, check=check)

def fix_system_calls_in_codebase():
    """
    Fix all # SECURITY: Replace with subprocess.run() - os.system() calls in the codebase
    This function scans and replaces unsafe patterns
    """

    logger.info("Starting security fixes for system calls...")

    src_path = Path('src')
    if not src_path.exists():
        logger.error("Source directory not found")
        return

    fixed_files = []

    for py_file in src_path.rglob('*.py'):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content

            # Pattern 1: Direct os.system calls
            content = re.sub(
                r"os\.system\s*\(\s*['\"]([^'\"]*)['\"]",
                r"secure_execute('\1')",
                content
            )

            # Pattern 2: os.system with variables
            content = re.sub(
                r"os\.system\s*\(\s*([^)]+)\)",
                r"secure_execute(\1)",
                content
            )

            # Add import if we made changes and import doesn't exist
            if content != original_content:
                if 'from infrastructure.secure_commands import secure_execute' not in content:
                    # Add import at the top after other imports
                    import_line = 'from infrastructure.secure_commands import secure_execute\n'

                    # Find the right place to insert (after existing imports)
                    lines = content.split('\n')
                    insert_index = 0

                    for i, line in enumerate(lines):
                        if line.strip().startswith('import ') or line.strip().startswith('from '):
                            insert_index = i + 1
                        elif line.strip() and not line.startswith('#'):
                            break

                    lines.insert(insert_index, import_line)
                    content = '\n'.join(lines)

                # Write the fixed content
                with open(py_file, 'w', encoding='utf-8') as f:
                    f.write(content)

                fixed_files.append(str(py_file))
                logger.info(f"Fixed system calls in: {py_file}")

        except Exception as e:
            logger.error(f"Error processing {py_file}: {e}")

    logger.info(f"Security fixes completed. Modified {len(fixed_files)} files:")
    for file in fixed_files:
        logger.info(f"  - {file}")

# Legacy compatibility functions
def replace_os_system():
    """Legacy function name for compatibility"""
    fix_system_calls_in_codebase()

if __name__ == "__main__":
    # Run the security fixes
    logging.basicConfig(level=logging.INFO)
    fix_system_calls_in_codebase()
    print("✅ Security fixes for system calls completed!")
    print("⚠️  Manual review recommended for complex cases")