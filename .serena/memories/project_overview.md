# BEV OSINT Framework - Project Overview

## Purpose
The BEV OSINT Framework is a comprehensive Open Source Intelligence (OSINT) platform designed for cybersecurity research and threat analysis. It integrates multiple components including IntelOwl, Cytoscape.js, Neo4j, and custom analyzers for intelligence gathering and visualization.

## Key Features
- **Single-user deployment** with no authentication for maximum performance
- **Dark theme** hacker aesthetic throughout the interface
- **Tor integration** with built-in SOCKS5 proxy and automatic circuit rotation
- **Custom analyzers** for breach database search, darknet market scraping, cryptocurrency tracking, and social media analysis
- **Graph visualization** using Cytoscape.js and Neo4j
- **Comprehensive monitoring** with Prometheus, Grafana, and custom metrics

## Target Use Cases
- Cybersecurity research and threat modeling
- OSINT investigations and intelligence gathering
- Academic security research
- Network topology analysis and visualization
- Breach database correlation and analysis

## Architecture Overview
The system is built on a microservices architecture with:
- **Frontend**: IntelOwl web interface with dark theme
- **Analyzers**: Custom OSINT analyzers for various data sources
- **Storage**: PostgreSQL, Neo4j, Redis, Elasticsearch
- **Message Queue**: RabbitMQ for job processing
- **Monitoring**: Prometheus + Grafana stack
- **Orchestration**: Apache Airflow for workflows
- **Proxy**: Tor integration for anonymized requests

## Important Security Note
This is a **RESEARCH FRAMEWORK** designed for:
- Authorized security research only
- Academic and educational purposes
- Private network deployment only
- Professional cybersecurity analysis

The system has NO AUTHENTICATION and should NEVER be exposed to public networks.