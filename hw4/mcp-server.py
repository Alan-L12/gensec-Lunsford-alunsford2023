# Author: Alan Lunsford (FAU ID: alunsford2023)
"""
Custom MCP Server (SecAudit MCP Server) for HW4.
Implements custom security auditing, text metrics, and prompt template tools over stdio.
"""

import re
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("SecAudit-Custom-Server")


@mcp.tool()
def scan_text_for_secrets(text: str) -> dict:
    """
    Scan input text for potential sensitive artifacts such as API keys, emails, or IPv4 addresses.

    Args:
        text: The raw text string to inspect for sensitive security patterns.

    Returns:
        A dictionary summarizing any detected patterns and a risk assessment status.
    """
    findings = {
        "emails_found": re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text),
        "ipv4_found": re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text),
        "api_key_patterns": re.findall(r"(?:sk-|AIza|ghp_)[a-zA-Z0-9_\-]{10,}", text),
    }
    total_hits = sum(len(v) for v in findings.values())
    findings["risk_level"] = "HIGH" if total_hits > 0 else "LOW"
    findings["total_findings"] = total_hits
    return findings


@mcp.tool()
def compute_audit_metrics(text: str) -> dict:
    """
    Compute structural metrics (character count, word count, line count) for audited content.

    Args:
        text: The text content to analyze.

    Returns:
        A dictionary containing character, word, and line counts.
    """
    lines = text.splitlines()
    words = text.split()
    return {
        "character_count": len(text),
        "word_count": len(words),
        "line_count": len(lines),
    }


@mcp.tool()
def format_security_report(target: str, risk_level: str, summary: str) -> str:
    """
    Generate a standardized security audit report string for a given target.

    Args:
        target: The URL or resource name that was audited.
        risk_level: Assessed risk level (e.g., LOW, MEDIUM, HIGH).
        summary: Key findings from the multi-step analysis.

    Returns:
        A formatted markdown audit report.
    """
    return (
        f"=== SECURITY AUDIT REPORT (Student: alunsford2023) ===\n"
        f"Target Resource : {target}\n"
        f"Risk Assessment : {risk_level}\n"
        f"Audit Summary   : {summary}\n"
        f"======================================================"
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
