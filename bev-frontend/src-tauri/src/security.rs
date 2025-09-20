// Security Manager - Content Sanitization and CSP Enforcement

use anyhow::Result;
use std::collections::HashSet;

pub struct SecurityManager {
    allowed_domains: HashSet<String>,
    blocked_patterns: Vec<String>,
}

impl SecurityManager {
    pub fn new() -> Self {
        let mut allowed_domains = HashSet::new();
        allowed_domains.insert("localhost".to_string());
        allowed_domains.insert("127.0.0.1".to_string());
        
        let blocked_patterns = vec![
            "javascript:".to_string(),
            "data:text/html".to_string(),
            "vbscript:".to_string(),
        ];
        
        SecurityManager {
            allowed_domains,
            blocked_patterns,
        }
    }
    
    pub fn validate_url(&self, url: &str) -> bool {
        // Block any suspicious patterns
        for pattern in &self.blocked_patterns {
            if url.contains(pattern) {
                return false;
            }
        }
        true
    }
    
    pub fn sanitize_content(&self, content: &str) -> String {
        // Basic XSS prevention - will integrate DOMPurify on frontend
        content
            .replace("<script", "&lt;script")
            .replace("</script>", "&lt;/script&gt;")
            .replace("onerror=", "data-error=")
            .replace("onclick=", "data-click=")
    }
}