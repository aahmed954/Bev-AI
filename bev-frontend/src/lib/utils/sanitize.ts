/**
 * Data Sanitization Utility for BEV OSINT Framework
 * Ensures all potentially hostile content is properly sanitized before display
 */

import DOMPurify from 'dompurify';

/**
 * Sanitize HTML content to prevent XSS attacks
 */
export function sanitizeHTML(dirty: string): string {
    return DOMPurify.sanitize(dirty, {
        ALLOWED_TAGS: ['b', 'i', 'em', 'strong', 'span', 'p', 'br'],
        ALLOWED_ATTR: ['class', 'style'],
        KEEP_CONTENT: true,
        SAFE_FOR_TEMPLATES: true,
    });
}

/**
 * Sanitize text content (removes all HTML)
 */
export function sanitizeText(dirty: string): string {
    return DOMPurify.sanitize(dirty, {
        ALLOWED_TAGS: [],
        ALLOWED_ATTR: [],
        KEEP_CONTENT: true,
    });
}

/**
 * Sanitize JSON data
 */
export function sanitizeJSON(data: any): any {
    if (typeof data === 'string') {
        return sanitizeText(data);
    }
    
    if (Array.isArray(data)) {
        return data.map(item => sanitizeJSON(item));
    }
    
    if (data !== null && typeof data === 'object') {
        const sanitized: any = {};
        for (const key in data) {
            if (data.hasOwnProperty(key)) {
                sanitized[key] = sanitizeJSON(data[key]);
            }
        }
        return sanitized;
    }
    
    return data;
}

/**
 * Sanitize URL to prevent javascript: and data: URLs
 */
export function sanitizeURL(url: string): string {
    const sanitized = sanitizeText(url);
    
    // Block dangerous protocols
    const dangerousProtocols = ['javascript:', 'data:', 'vbscript:', 'file:'];
    const lowerURL = sanitized.toLowerCase();
    
    for (const protocol of dangerousProtocols) {
        if (lowerURL.startsWith(protocol)) {
            return '#';
        }
    }
    
    return sanitized;
}

/**
 * Sanitize crypto address (strict format validation)
 */
export function sanitizeCryptoAddress(address: string): string {
    // Remove any HTML/scripts
    const clean = sanitizeText(address);
    
    // Basic validation patterns for common crypto addresses
    const patterns = {
        btc: /^[13][a-km-zA-HJ-NP-Z1-9]{25,34}$/,
        eth: /^0x[a-fA-F0-9]{40}$/,
        xmr: /^4[0-9AB][1-9A-HJ-NP-Za-km-z]{93}$/,
    };
    
    // Check if it matches any known pattern
    const isValid = Object.values(patterns).some(pattern => pattern.test(clean));
    
    // Return sanitized address or empty string if invalid
    return isValid ? clean : '';
}

/**
 * Sanitize IOC (Indicator of Compromise) values
 */
export function sanitizeIOC(ioc: string, type: string): string {
    const clean = sanitizeText(ioc);
    
    switch (type) {
        case 'ip':
            // IPv4 pattern
            const ipv4Pattern = /^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$/;
            // IPv6 pattern (simplified)
            const ipv6Pattern = /^(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}$/;
            return (ipv4Pattern.test(clean) || ipv6Pattern.test(clean)) ? clean : '';
            
        case 'domain':
            // Domain pattern
            const domainPattern = /^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z0-9][a-z0-9-]{0,61}[a-z0-9]$/i;
            return domainPattern.test(clean) ? clean : '';
            
        case 'hash':
            // MD5, SHA1, SHA256 patterns
            const hashPatterns = [
                /^[a-fA-F0-9]{32}$/,  // MD5
                /^[a-fA-F0-9]{40}$/,  // SHA1
                /^[a-fA-F0-9]{64}$/,  // SHA256
            ];
            return hashPatterns.some(p => p.test(clean)) ? clean : '';
            
        case 'email':
            // Basic email pattern
            const emailPattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
            return emailPattern.test(clean) ? clean : '';
            
        case 'url':
            return sanitizeURL(clean);
            
        default:
            return clean;
    }
}

/**
 * Sanitize file path (prevent directory traversal)
 */
export function sanitizeFilePath(path: string): string {
    const clean = sanitizeText(path);
    
    // Remove directory traversal attempts
    const sanitized = clean
        .replace(/\.\./g, '')
        .replace(/\/\//g, '/')
        .replace(/\\\\/g, '\\');
    
    return sanitized;
}

/**
 * Sanitize search query
 */
export function sanitizeSearchQuery(query: string): string {
    // Remove HTML and limit length
    const clean = sanitizeText(query);
    return clean.substring(0, 200);
}

/**
 * Batch sanitize an array of items
 */
export function sanitizeArray<T>(items: T[], sanitizer: (item: T) => T): T[] {
    return items.map(item => sanitizer(item));
}

/**
 * Create a safe ID from a string (for DOM IDs)
 */
export function createSafeId(str: string): string {
    const clean = sanitizeText(str);
    return clean
        .toLowerCase()
        .replace(/[^a-z0-9]/g, '-')
        .replace(/^-+|-+$/g, '')
        .substring(0, 50);
}

/**
 * Validate and sanitize numeric input
 */
export function sanitizeNumber(value: any, min?: number, max?: number): number {
    const num = parseFloat(value);
    
    if (isNaN(num)) {
        return min ?? 0;
    }
    
    if (min !== undefined && num < min) {
        return min;
    }
    
    if (max !== undefined && num > max) {
        return max;
    }
    
    return num;
}

/**
 * Deep clone and sanitize an object
 */
export function deepSanitize<T>(obj: T): T {
    return sanitizeJSON(JSON.parse(JSON.stringify(obj)));
}

/**
 * Export all sanitizers as a single object for convenience
 */
export const sanitize = {
    html: sanitizeHTML,
    text: sanitizeText,
    json: sanitizeJSON,
    url: sanitizeURL,
    cryptoAddress: sanitizeCryptoAddress,
    ioc: sanitizeIOC,
    filePath: sanitizeFilePath,
    searchQuery: sanitizeSearchQuery,
    array: sanitizeArray,
    safeId: createSafeId,
    number: sanitizeNumber,
    deep: deepSanitize,
};

export default sanitize;