#!/usr/bin/env python3
"""
Fix wildcard imports across BEV OSINT Framework codebase
Replaces dangerous 'from X import *' with explicit imports
"""

import os
import ast
import re
from pathlib import Path
from typing import List, Dict, Set
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WildcardImportFixer:
    """
    Analyzes and fixes wildcard imports in Python files
    """

    def __init__(self):
        self.fixed_files = []
        self.analysis_results = {}

    def analyze_file(self, filepath: Path) -> Dict[str, List[str]]:
        """
        Analyze a Python file for wildcard imports

        Returns:
            Dictionary with 'wildcard_imports' and 'issues' keys
        """
        result = {
            'wildcard_imports': [],
            'issues': [],
            'suggested_fixes': []
        }

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # Parse AST to find imports
            try:
                tree = ast.parse(content)
            except SyntaxError as e:
                result['issues'].append(f"Syntax error: {e}")
                return result

            # Find wildcard imports
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.names and any(alias.name == '*' for alias in node.names):
                        module_name = node.module or "relative_import"
                        result['wildcard_imports'].append({
                            'module': module_name,
                            'line': node.lineno,
                            'level': node.level
                        })

            # Suggest specific fixes for known problematic imports
            lines = content.split('\n')
            for i, line in enumerate(lines, 1):
                if 'from anticaptchaofficial.recaptchav2proxyless import *' in line:
                    result['suggested_fixes'].append({
                        'line': i,
                        'original': line.strip(),
                        'suggested': 'from anticaptchaofficial.recaptchav2proxyless import RecaptchaV2Proxyless'
                    })

                if 'from anticaptchaofficial.funcaptchaproxyless import *' in line:
                    result['suggested_fixes'].append({
                        'line': i,
                        'original': line.strip(),
                        'suggested': 'from anticaptchaofficial.funcaptchaproxyless import FuncaptchaProxyless'
                    })

        except Exception as e:
            result['issues'].append(f"Error analyzing file: {e}")

        return result

    def fix_known_wildcards(self, filepath: Path) -> bool:
        """
        Fix known problematic wildcard imports with specific replacements
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content
            modified = False

            # Known fixes for anticaptcha imports
            replacements = {
                'from anticaptchaofficial.recaptchav2proxyless import *':
                    'from anticaptchaofficial.recaptchav2proxyless import RecaptchaV2Proxyless',

                'from anticaptchaofficial.funcaptchaproxyless import *':
                    'from anticaptchaofficial.funcaptchaproxyless import FuncaptchaProxyless',

                # Add more specific replacements as needed
                'from requests import *':
                    'import requests',

                'from json import *':
                    'import json',

                'from datetime import *':
                    'from datetime import datetime, timedelta, timezone',
            }

            for old_import, new_import in replacements.items():
                if old_import in content:
                    content = content.replace(old_import, new_import)
                    modified = True
                    logger.info(f"Fixed wildcard import in {filepath}: {old_import}")

            # For other wildcard imports, add warning comments
            wildcard_pattern = r'from\s+([^\s]+)\s+import\s+\*'
            def add_warning(match):
                module = match.group(1)
                original_line = match.group(0)
                return f"# SECURITY WARNING: Wildcard import detected - replace with explicit imports\n# {original_line}\n# from {module} import SpecificClass1, SpecificClass2\n{original_line}"

            if re.search(wildcard_pattern, content):
                content = re.sub(wildcard_pattern, add_warning, content)
                modified = True

            if modified:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True

        except Exception as e:
            logger.error(f"Error fixing {filepath}: {e}")

        return False

    def scan_codebase(self, src_path: Path = None) -> Dict[str, Dict]:
        """
        Scan entire codebase for wildcard imports
        """
        if src_path is None:
            src_path = Path('src')

        if not src_path.exists():
            logger.error(f"Source directory not found: {src_path}")
            return {}

        results = {}

        for py_file in src_path.rglob('*.py'):
            if py_file.name.startswith('.'):
                continue

            analysis = self.analyze_file(py_file)
            if analysis['wildcard_imports'] or analysis['issues'] or analysis['suggested_fixes']:
                results[str(py_file)] = analysis

        return results

    def fix_all_wildcards(self, src_path: Path = None):
        """
        Fix all wildcard imports in the codebase
        """
        if src_path is None:
            src_path = Path('src')

        logger.info("Scanning for wildcard imports...")
        results = self.scan_codebase(src_path)

        if not results:
            logger.info("No wildcard imports found!")
            return

        logger.info(f"Found wildcard imports in {len(results)} files")

        fixed_count = 0
        for filepath_str, analysis in results.items():
            filepath = Path(filepath_str)

            if analysis['wildcard_imports']:
                logger.info(f"Processing: {filepath}")

                if self.fix_known_wildcards(filepath):
                    fixed_count += 1
                    self.fixed_files.append(str(filepath))

        logger.info(f"Fixed wildcard imports in {fixed_count} files")

        # Generate report
        self.generate_report(results)

    def generate_report(self, results: Dict[str, Dict]):
        """
        Generate a detailed report of wildcard import issues
        """
        report_content = []
        report_content.append("# Wildcard Import Security Analysis Report")
        report_content.append("=" * 50)
        report_content.append("")

        total_files = len(results)
        total_wildcards = sum(len(data['wildcard_imports']) for data in results.values())

        report_content.append(f"**Summary:**")
        report_content.append(f"- Files with issues: {total_files}")
        report_content.append(f"- Total wildcard imports: {total_wildcards}")
        report_content.append(f"- Files fixed: {len(self.fixed_files)}")
        report_content.append("")

        for filepath, analysis in results.items():
            report_content.append(f"## {filepath}")
            report_content.append("")

            if analysis['wildcard_imports']:
                report_content.append("**Wildcard Imports:**")
                for imp in analysis['wildcard_imports']:
                    report_content.append(f"- Line {imp['line']}: from {imp['module']} import *")

            if analysis['suggested_fixes']:
                report_content.append("**Suggested Fixes:**")
                for fix in analysis['suggested_fixes']:
                    report_content.append(f"- Line {fix['line']}:")
                    report_content.append(f"  - Original: `{fix['original']}`")
                    report_content.append(f"  - Suggested: `{fix['suggested']}`")

            if analysis['issues']:
                report_content.append("**Issues:**")
                for issue in analysis['issues']:
                    report_content.append(f"- {issue}")

            report_content.append("")

        # Write report
        report_path = Path('wildcard_import_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_content))

        logger.info(f"Report generated: {report_path}")

def main():
    """Main execution function"""
    fixer = WildcardImportFixer()

    print("🔍 BEV Wildcard Import Security Fixer")
    print("=" * 40)

    # Scan and fix
    fixer.fix_all_wildcards()

    print("\n✅ Wildcard import security fixes completed!")
    print(f"📄 Report generated: wildcard_import_report.md")

    if fixer.fixed_files:
        print(f"🔧 Fixed {len(fixer.fixed_files)} files:")
        for file in fixer.fixed_files:
            print(f"   - {file}")
    else:
        print("ℹ️  No automatic fixes applied - manual review required")

    print("\n⚠️  Next steps:")
    print("1. Review wildcard_import_report.md")
    print("2. Replace remaining wildcard imports manually")
    print("3. Test imports work correctly")
    print("4. Update any usage of imported classes")

if __name__ == "__main__":
    main()