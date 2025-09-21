"""Metadata Analyzer for IntelOwl
Extracts and analyzes metadata from files
Integrates with existing metadata_scrubber.py
"""

import os
import sys
import json
from typing import Dict, List, Any
from datetime import datetime
import hashlib
from celery import shared_task
from api_app.analyzers_manager.file_analyzers import FileAnalyzer
import logging
import exifread
from PIL import Image
from PyPDF2 import PdfReader
import python_docx
from mutagen import File as MutagenFile

# Import our existing metadata scrubber
sys.path.append('/opt/bev_src/enhancement')
from metadata_scrubber import MetadataScrubber

logger = logging.getLogger(__name__)


class MetadataAnalyzer(FileAnalyzer):
    """Extract and analyze file metadata"""
    
    def set_params(self, params):
        """Set analyzer parameters"""
        self.deep_analysis = params.get('deep_analysis', True)
        self.extract_gps = params.get('extract_gps', True)
        self.extract_author = params.get('extract_author', True)
        self.extract_software = params.get('extract_software', True)
        self.check_steganography = params.get('check_steganography', False)
        
    def run(self):
        """Execute metadata analysis"""
        file_path = self.filepath
        results = {
            'metadata': {},
            'sensitive_info': [],
            'gps_location': None,
            'author_info': {},
            'software_used': [],
            'creation_timeline': {},
            'privacy_risks': [],
            'hidden_data': []
        }
        
        try:
            # Determine file type
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']:
                results.update(self._analyze_image(file_path))
                
            elif file_ext == '.pdf':
                results.update(self._analyze_pdf(file_path))
                
            elif file_ext in ['.doc', '.docx']:
                results.update(self._analyze_document(file_path))
                
            elif file_ext in ['.mp3', '.mp4', '.avi', '.mov', '.wav']:
                results.update(self._analyze_media(file_path))
                
            else:
                # Generic metadata extraction
                results['metadata'] = self._extract_generic_metadata(file_path)
                
            # Use our existing metadata scrubber for deep analysis
            if self.deep_analysis:
                scrubber = MetadataScrubber()
                deep_metadata = scrubber.extract_all_metadata(file_path)
                results['metadata'].update(deep_metadata)
                
            # Assess privacy risks
            results['privacy_risks'] = self._assess_privacy_risks(results)
            
            # Check for steganography
            if self.check_steganography and file_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                results['hidden_data'] = self._check_steganography(file_path)
                
        except Exception as e:
            logger.error(f"Metadata analysis failed: {str(e)}")
            return {'error': str(e)}
            
        return results
        
    def _analyze_image(self, file_path: str) -> Dict:
        """Analyze image metadata"""
        results = {
            'metadata': {},
            'gps_location': None,
            'author_info': {},
            'software_used': []
        }
        
        try:
            # Extract EXIF data
            with open(file_path, 'rb') as f:
                tags = exifread.process_file(f, details=True)
                
                for tag, value in tags.items():
                    results['metadata'][str(tag)] = str(value)
                    
            # Extract GPS coordinates
            if self.extract_gps:
                gps_data = self._extract_gps_from_exif(tags)
                if gps_data:
                    results['gps_location'] = gps_data
                    results['sensitive_info'].append({
                        'type': 'GPS_LOCATION',
                        'value': gps_data,
                        'risk': 'HIGH'
                    })
                    
            # Extract author/copyright
            if self.extract_author:
                author_tags = ['Image Artist', 'Image Copyright', 'Image XPAuthor']
                for tag in author_tags:
                    if tag in tags:
                        results['author_info'][tag] = str(tags[tag])
                        results['sensitive_info'].append({
                            'type': 'AUTHOR',
                            'value': str(tags[tag]),
                            'risk': 'MEDIUM'
                        })
                        
            # Extract software info
            if self.extract_software:
                software_tags = ['Image Software', 'Image ProcessingSoftware']
                for tag in software_tags:
                    if tag in tags:
                        results['software_used'].append(str(tags[tag]))
                        
            # Extract dates
            date_tags = ['EXIF DateTimeOriginal', 'EXIF DateTimeDigitized', 'Image DateTime']
            for tag in date_tags:
                if tag in tags:
                    results['creation_timeline'][tag] = str(tags[tag])
                    
            # Use PIL for additional metadata
            img = Image.open(file_path)
            if hasattr(img, '_getexif') and img._getexif():
                exif_data = img._getexif()
                results['metadata']['dimensions'] = f"{img.width}x{img.height}"
                results['metadata']['format'] = img.format
                results['metadata']['mode'] = img.mode
                
        except Exception as e:
            logger.error(f"Image analysis failed: {str(e)}")
            
        return results
        
    def _analyze_pdf(self, file_path: str) -> Dict:
        """Analyze PDF metadata"""
        results = {
            'metadata': {},
            'author_info': {},
            'software_used': []
        }
        
        try:
            with open(file_path, 'rb') as f:
                pdf = PdfReader(f)
                
                # Extract metadata
                metadata = pdf.metadata
                if metadata:
                    for key, value in metadata.items():
                        results['metadata'][key] = str(value)
                        
                    # Extract author
                    if '/Author' in metadata:
                        results['author_info']['author'] = str(metadata['/Author'])
                        results['sensitive_info'].append({
                            'type': 'AUTHOR',
                            'value': str(metadata['/Author']),
                            'risk': 'MEDIUM'
                        })
                        
                    # Extract software
                    if '/Producer' in metadata:
                        results['software_used'].append(str(metadata['/Producer']))
                    if '/Creator' in metadata:
                        results['software_used'].append(str(metadata['/Creator']))
                        
                    # Extract dates
                    if '/CreationDate' in metadata:
                        results['creation_timeline']['created'] = str(metadata['/CreationDate'])
                    if '/ModDate' in metadata:
                        results['creation_timeline']['modified'] = str(metadata['/ModDate'])
                        
                # Extract text statistics
                num_pages = len(pdf.pages)
                results['metadata']['page_count'] = num_pages
                
                # Check for JavaScript (security risk)
                if '/JavaScript' in pdf.trailer or '/JS' in pdf.trailer:
                    results['sensitive_info'].append({
                        'type': 'JAVASCRIPT',
                        'value': 'PDF contains JavaScript',
                        'risk': 'HIGH'
                    })
                    
        except Exception as e:
            logger.error(f"PDF analysis failed: {str(e)}")
            
        return results
        
    def _analyze_document(self, file_path: str) -> Dict:
        """Analyze Word document metadata"""
        results = {
            'metadata': {},
            'author_info': {},
            'software_used': []
        }
        
        try:
            doc = python_docx.Document(file_path)
            
            # Extract core properties
            props = doc.core_properties
            
            if props.author:
                results['author_info']['author'] = props.author
                results['sensitive_info'].append({
                    'type': 'AUTHOR',
                    'value': props.author,
                    'risk': 'MEDIUM'
                })
                
            if props.last_modified_by:
                results['author_info']['last_modified_by'] = props.last_modified_by
                
            if props.title:
                results['metadata']['title'] = props.title
                
            if props.subject:
                results['metadata']['subject'] = props.subject
                
            if props.keywords:
                results['metadata']['keywords'] = props.keywords
                
            if props.comments:
                results['metadata']['comments'] = props.comments
                results['sensitive_info'].append({
                    'type': 'COMMENTS',
                    'value': props.comments,
                    'risk': 'LOW'
                })
                
            # Extract dates
            if props.created:
                results['creation_timeline']['created'] = props.created.isoformat()
            if props.modified:
                results['creation_timeline']['modified'] = props.modified.isoformat()
            if props.last_printed:
                results['creation_timeline']['last_printed'] = props.last_printed.isoformat()
                
            # Extract revision number
            if props.revision:
                results['metadata']['revision'] = props.revision
                
            # Word count and stats
            results['metadata']['paragraph_count'] = len(doc.paragraphs)
            
        except Exception as e:
            logger.error(f"Document analysis failed: {str(e)}")
            
        return results
        
    def _analyze_media(self, file_path: str) -> Dict:
        """Analyze media file metadata"""
        results = {
            'metadata': {},
            'author_info': {},
            'software_used': []
        }
        
        try:
            media = MutagenFile(file_path)
            
            if media and media.tags:
                for key, value in media.tags.items():
                    results['metadata'][key] = str(value)
                    
                # Extract common tags
                if 'artist' in media.tags:
                    results['author_info']['artist'] = str(media.tags['artist'])
                    
                if 'album' in media.tags:
                    results['metadata']['album'] = str(media.tags['album'])
                    
                if 'date' in media.tags:
                    results['creation_timeline']['date'] = str(media.tags['date'])
                    
                if 'copyright' in media.tags:
                    results['author_info']['copyright'] = str(media.tags['copyright'])
                    results['sensitive_info'].append({
                        'type': 'COPYRIGHT',
                        'value': str(media.tags['copyright']),
                        'risk': 'MEDIUM'
                    })
                    
            # Extract technical info
            if media and media.info:
                results['metadata']['length'] = media.info.length
                results['metadata']['bitrate'] = media.info.bitrate
                
                if hasattr(media.info, 'channels'):
                    results['metadata']['channels'] = media.info.channels
                if hasattr(media.info, 'sample_rate'):
                    results['metadata']['sample_rate'] = media.info.sample_rate
                    
        except Exception as e:
            logger.error(f"Media analysis failed: {str(e)}")
            
        return results
        
    def _extract_generic_metadata(self, file_path: str) -> Dict:
        """Extract generic file metadata"""
        metadata = {}
        
        try:
            stat_info = os.stat(file_path)
            
            metadata['file_size'] = stat_info.st_size
            metadata['created'] = datetime.fromtimestamp(stat_info.st_ctime).isoformat()
            metadata['modified'] = datetime.fromtimestamp(stat_info.st_mtime).isoformat()
            metadata['accessed'] = datetime.fromtimestamp(stat_info.st_atime).isoformat()
            metadata['permissions'] = oct(stat_info.st_mode)
            
            # Calculate file hash
            with open(file_path, 'rb') as f:
                metadata['md5'] = hashlib.md5(f.read()).hexdigest()
                f.seek(0)
                metadata['sha256'] = hashlib.sha256(f.read()).hexdigest()
                
        except Exception as e:
            logger.error(f"Generic metadata extraction failed: {str(e)}")
            
        return metadata
        
    def _extract_gps_from_exif(self, tags: Dict) -> Optional[Dict]:
        """Extract GPS coordinates from EXIF tags"""
        gps_data = {}
        
        # GPS tags to look for
        gps_tags = {
            'GPS GPSLatitude': 'latitude',
            'GPS GPSLatitudeRef': 'latitude_ref',
            'GPS GPSLongitude': 'longitude',
            'GPS GPSLongitudeRef': 'longitude_ref',
            'GPS GPSAltitude': 'altitude',
            'GPS GPSAltitudeRef': 'altitude_ref'
        }
        
        for tag, key in gps_tags.items():
            if tag in tags:
                gps_data[key] = str(tags[tag])
                
        if 'latitude' in gps_data and 'longitude' in gps_data:
            # Convert to decimal degrees
            lat = self._convert_to_degrees(gps_data['latitude'])
            lon = self._convert_to_degrees(gps_data['longitude'])
            
            if gps_data.get('latitude_ref') == 'S':
                lat = -lat
            if gps_data.get('longitude_ref') == 'W':
                lon = -lon
                
            return {
                'latitude': lat,
                'longitude': lon,
                'altitude': gps_data.get('altitude'),
                'coordinates': f"{lat}, {lon}"
            }
            
        return None
        
    def _convert_to_degrees(self, value: str) -> float:
        """Convert GPS coordinates to decimal degrees"""
        # Value format: '[34, 12, 5067/100]'
        parts = value.strip('[]').split(', ')
        if len(parts) == 3:
            degrees = float(parts[0])
            minutes = float(parts[1])
            
            # Handle fraction
            if '/' in parts[2]:
                sec_parts = parts[2].split('/')
                seconds = float(sec_parts[0]) / float(sec_parts[1])
            else:
                seconds = float(parts[2])
                
            return degrees + minutes/60 + seconds/3600
            
        return 0.0
        
    def _assess_privacy_risks(self, results: Dict) -> List[Dict]:
        """Assess privacy and security risks from metadata"""
        risks = []
        
        # GPS location is high risk
        if results.get('gps_location'):
            risks.append({
                'type': 'LOCATION_DISCLOSURE',
                'description': 'GPS coordinates found in metadata',
                'severity': 'HIGH',
                'recommendation': 'Remove GPS data before sharing'
            })
            
        # Author information
        if results.get('author_info'):
            risks.append({
                'type': 'IDENTITY_DISCLOSURE',
                'description': 'Author/creator information found',
                'severity': 'MEDIUM',
                'recommendation': 'Remove author metadata'
            })
            
        # Software versions can reveal vulnerabilities
        if results.get('software_used'):
            risks.append({
                'type': 'SOFTWARE_DISCLOSURE',
                'description': 'Software version information exposed',
                'severity': 'LOW',
                'recommendation': 'Consider removing software metadata'
            })
            
        # Old creation dates might reveal timeline
        timeline = results.get('creation_timeline', {})
        if timeline:
            risks.append({
                'type': 'TIMELINE_DISCLOSURE',
                'description': 'Creation and modification dates exposed',
                'severity': 'LOW',
                'recommendation': 'Sanitize date metadata if sensitive'
            })
            
        return risks
        
    def _check_steganography(self, file_path: str) -> List[Dict]:
        """Check for hidden data using steganography"""
        hidden_data = []
        
        # This is a placeholder - real implementation would use
        # tools like steghide, stegsolve, zsteg, etc.
        
        return hidden_data
        
    @classmethod
    def _monkeypatch(cls):
        """Register analyzer with IntelOwl"""
        patches = [
            {
                'model': 'analyzers_manager.AnalyzerConfig',
                'name': 'MetadataAnalyzer',
                'description': 'Extract and analyze file metadata',
                'python_module': 'custom_analyzers.metadata_analyzer.MetadataAnalyzer',
                'disabled': False,
                'type': 'file',
                'docker_based': False,
                'maximum_tlp': 'RED',
                'observable_supported': [],
                'supported_filetypes': [
                    'image/jpeg', 'image/png', 'image/gif', 'image/bmp',
                    'application/pdf', 'application/msword',
                    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                    'audio/mpeg', 'video/mp4', 'audio/wav'
                ],
                'run_hash': False,
                'run_hash_type': '',
                'not_supported_filetypes': [],
                'parameters': {
                    'deep_analysis': {
                        'type': 'bool',
                        'description': 'Perform deep metadata analysis',
                        'default': True
                    },
                    'extract_gps': {
                        'type': 'bool',
                        'description': 'Extract GPS coordinates',
                        'default': True
                    },
                    'check_steganography': {
                        'type': 'bool',
                        'description': 'Check for hidden data',
                        'default': False
                    }
                }
            }
        ]
        return patches
