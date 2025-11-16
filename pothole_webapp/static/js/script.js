// Professional UI interactions
let currentResultsData = null;

document.addEventListener('DOMContentLoaded', function() {
    initializePresetButtons();
    initializeProgressAnimation();
    initializeLocationFeatures();
});
// Check if image URL is accessible

// Preset button functionality
function initializePresetButtons() {
    const presetButtons = document.querySelectorAll('.preset-btn');
    presetButtons.forEach(button => {
        button.addEventListener('click', function() {
            const targetId = this.getAttribute('data-target');
            const value = this.getAttribute('data-value');
            document.getElementById(targetId).value = value;
            
            // Update active state
            presetButtons.forEach(btn => btn.classList.remove('active'));
            this.classList.add('active');
        });
    });
}

// Progress animation
function initializeProgressAnimation() {
    // This will be used during loading
}

// Location features initialization
function initializeLocationFeatures() {
    const getLocationBtn = document.getElementById('getCurrentLocation');
    
    if (getLocationBtn) {
        getLocationBtn.addEventListener('click', getCurrentLocation);
    }
}

// Get current location using Geolocation API
function getCurrentLocation() {
    const latitudeInput = document.getElementById('latitude');
    const longitudeInput = document.getElementById('longitude');
    const getLocationBtn = document.getElementById('getCurrentLocation');
    
    if (!navigator.geolocation) {
        showNotification('Geolocation is not supported by your browser', 'error');
        return;
    }
    
    // Show loading state
    const originalHTML = getLocationBtn.innerHTML;
    getLocationBtn.innerHTML = '<i class="fas fa-spinner location-btn-loading"></i>';
    getLocationBtn.disabled = true;
    
    navigator.geolocation.getCurrentPosition(
        function(position) {
            const lat = position.coords.latitude;
            const lng = position.coords.longitude;
            
            latitudeInput.value = lat.toFixed(6);
            longitudeInput.value = lng.toFixed(6);
            
            // Reverse geocoding to get address (optional)
            getAddressFromCoordinates(lat, lng);
            
            // Restore button
            getLocationBtn.innerHTML = originalHTML;
            getLocationBtn.disabled = false;
            
            showNotification('Location obtained successfully!', 'success');
        },
        function(error) {
            console.error('Error getting location:', error);
            
            let errorMessage = 'Unable to get your location. ';
            switch(error.code) {
                case error.PERMISSION_DENIED:
                    errorMessage += 'Please allow location access.';
                    break;
                case error.POSITION_UNAVAILABLE:
                    errorMessage += 'Location information unavailable.';
                    break;
                case error.TIMEOUT:
                    errorMessage += 'Location request timed out.';
                    break;
                default:
                    errorMessage += 'An unknown error occurred.';
                    break;
            }
            
            showNotification(errorMessage, 'error');
            
            // Restore button
            getLocationBtn.innerHTML = originalHTML;
            getLocationBtn.disabled = false;
        },
        {
            enableHighAccuracy: true,
            timeout: 10000,
            maximumAge: 60000
        }
    );
}

// Optional: Reverse geocoding to get address from coordinates
function getAddressFromCoordinates(lat, lng) {
    // Using OpenStreetMap Nominatim API (free)
    fetch(`https://nominatim.openstreetmap.org/reverse?format=json&lat=${lat}&lon=${lng}`)
        .then(response => response.json())
        .then(data => {
            if (data && data.address) {
                const address = data.address;
                
                // Auto-fill city if available
                const cityInput = document.getElementById('city');
                if (!cityInput.value) {
                    if (address.city) cityInput.value = address.city;
                    else if (address.town) cityInput.value = address.town;
                    else if (address.village) cityInput.value = address.village;
                }
                
                // Auto-fill location name if available
                const locationNameInput = document.getElementById('location_name');
                if (!locationNameInput.value && address.road) {
                    locationNameInput.value = address.road;
                }
            }
        })
        .catch(error => {
            console.log('Reverse geocoding failed:', error);
        });
}

// File upload handling with drag & drop
const fileInput = document.getElementById('fileInput');
const uploadBox = document.getElementById('uploadBox');
const filePreview = document.getElementById('filePreview');
const fileName = document.getElementById('fileName');
const fileSize = document.getElementById('fileSize');
const fileIcon = document.getElementById('fileIcon');

// Drag and drop functionality
uploadBox.addEventListener('click', () => fileInput.click());

uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.classList.add('dragover');
});

uploadBox.addEventListener('dragleave', () => {
    uploadBox.classList.remove('dragover');
});

uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.classList.remove('dragover');
    
    if (e.dataTransfer.files.length) {
        fileInput.files = e.dataTransfer.files;
        handleFileSelect();
    }
});

fileInput.addEventListener('change', handleFileSelect);

function handleFileSelect() {
    const file = fileInput.files[0];
    if (file) {
        const fileType = file.type.split('/')[0];
        const icon = fileType === 'video' ? 'fa-video' : 'fa-image';
        fileIcon.innerHTML = `<i class="fas ${icon}"></i>`;
        fileName.textContent = file.name;
        fileSize.textContent = formatFileSize(file.size);
        document.getElementById('fileType').textContent = fileType.toUpperCase();
        filePreview.classList.remove('hidden');
        uploadBox.classList.add('hidden');
    }
}

function clearFile() {
    fileInput.value = '';
    filePreview.classList.add('hidden');
    uploadBox.classList.remove('hidden');
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Form submission with enhanced loading
document.getElementById('uploadForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const formData = new FormData(this);
    const analyzeBtn = document.getElementById('analyzeBtn');
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');
    const error = document.getElementById('error');
    
    // Show loading, hide previous results/errors
    loading.classList.remove('hidden');
    results.classList.add('hidden');
    error.classList.add('hidden');
    analyzeBtn.disabled = true;
    
    // Update loading text based on file type
    const file = fileInput.files[0];
    const fileType = file?.type.startsWith('video/') ? 'video' : 'image';
    document.getElementById('loadingText').textContent = 
        `Processing ${fileType} for comprehensive road analysis...`;
    
    // Simulate progress for better UX
    simulateProgress();
    
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            currentResultsData = data; // Store for PDF export
            displayResults(data);
        } else {
            displayError(data.error);
        }
    } catch (error) {
        displayError('Upload failed: ' + error.message);
    } finally {
        loading.classList.add('hidden');
        analyzeBtn.disabled = false;
    }
});

function simulateProgress() {
    const progressFill = document.getElementById('progressFill');
    const progressPercentage = document.querySelector('.progress-percentage');
    let progress = 0;
    
    const interval = setInterval(() => {
        progress += Math.random() * 10;
        if (progress > 90) {
            progress = 90; // Hold at 90% until actual completion
        }
        progressFill.style.width = progress + '%';
        progressPercentage.textContent = Math.floor(progress) + '%';
    }, 200);
    
    // Store interval ID to clear later
    window.progressInterval = interval;
}

function displayResults(data) {
    // Clear progress simulation
    if (window.progressInterval) {
        clearInterval(window.progressInterval);
    }
    
    const resultsDiv = document.getElementById('results');
    const errorDiv = document.getElementById('error');
    
    errorDiv.classList.add('hidden');
    
    const isVideo = data.file_type === 'video';
    
    let resultImageHtml = '';
    if (data.result_image) {
        resultImageHtml = `
            <div class="section-card">
                <div class="card-header">
                    <i class="fas fa-map-marked-alt card-icon"></i>
                    <h2>Detected Potholes Visualization</h2>
                    <p>AI-identified potholes with dimension analysis</p>
                </div>
                <div class="result-image">
                    <img src="${data.result_image}" alt="Analysis Result" 
                         style="width: 100%; border-radius: 12px; border: 2px solid var(--border);"
                         onerror="this.style.display='none'; console.log('Image failed to load:', this.src)">
                    <div class="image-info">
                        <p><i class="fas fa-link"></i> Cloudinary URL: ${data.result_image}</p>
                    </div>
                </div>
            </div>
        `;
    } else {
        resultImageHtml = `
            <div class="section-card">
                <div class="card-header">
                    <i class="fas fa-map-marked-alt card-icon"></i>
                    <h2>Detected Potholes Visualization</h2>
                    <p>No visualization available for this analysis</p>
                </div>
                <div class="no-image">
                    <i class="fas fa-image" style="font-size: 3rem; color: var(--text-muted); margin-bottom: 1rem;"></i>
                    <p>No annotated image available</p>
                </div>
            </div>
        `;
    }
    
    let videoInfoHtml = '';
    if (isVideo && data.total_frames_analyzed) {
        videoInfoHtml = `
            <div class="stat-card">
                <div class="stat-icon">
                    <i class="fas fa-film"></i>
                </div>
                <span class="stat-number">${data.total_frames_analyzed}</span>
                <span class="stat-label">Frames Analyzed</span>
            </div>
        `;
    }
    
    // Format numbers safely
    const materialLiters = data.cost_breakdown?.material_required_liters?.toFixed(0) || '0';
    const totalHours = data.cost_breakdown?.time_breakdown?.total_hours?.toFixed(1) || '0';
    const materialCost = data.cost_breakdown?.material_cost?.toFixed(2) || '0.00';
    const laborCost = data.cost_breakdown?.labor_cost?.toFixed(2) || '0.00';
    const equipmentTransport = ((data.cost_breakdown?.equipment_cost || 0) + (data.cost_breakdown?.transport_cost || 0)).toFixed(2);
    const overheadCost = data.cost_breakdown?.overhead_cost?.toFixed(2) || '0.00';
    const totalCost = data.cost_breakdown?.total_cost?.toFixed(2) || '0.00';
    const overheadPercentage = data.cost_breakdown?.overhead_percentage || '0';
    
    // Location information
    const locationData = data.location_data || {};
    let locationHtml = '';
    if (locationData.location_name || locationData.city) {
        locationHtml = `
            <div class="location-info-card">
                <h3 style="margin-bottom: 1.5rem; display: flex; align-items: center; gap: 0.75rem;">
                    <i class="fas fa-map-marker-alt" style="color: var(--primary-light);"></i>
                    Location Information
                </h3>
                <div class="location-details">
                    ${locationData.location_name ? `
                        <div class="location-detail-item">
                            <span class="location-detail-label">Road Name</span>
                            <span class="location-detail-value">${locationData.location_name}</span>
                        </div>
                    ` : ''}
                    
                    ${locationData.city ? `
                        <div class="location-detail-item">
                            <span class="location-detail-label">City</span>
                            <span class="location-detail-value">${locationData.city}</span>
                        </div>
                    ` : ''}
                    
                    ${locationData.latitude && locationData.longitude ? `
                        <div class="location-detail-item">
                            <span class="location-detail-label">Coordinates</span>
                            <span class="location-detail-value">
                                <a href="https://maps.google.com/?q=${locationData.latitude},${locationData.longitude}" 
                                   target="_blank" class="map-link">
                                    ${parseFloat(locationData.latitude).toFixed(4)}, ${parseFloat(locationData.longitude).toFixed(4)}
                                    <i class="fas fa-external-link-alt"></i>
                                </a>
                            </span>
                        </div>
                    ` : ''}
                </div>
            </div>
        `;
    }
    
    resultsDiv.innerHTML = `
        <div class="results-header">
            <h2><i class="fas fa-check-circle"></i> Analysis Complete</h2>
            <p>${isVideo ? 'Video analysis finished successfully' : 'Image analysis finished successfully'}</p>
        </div>
        
        ${locationHtml}
        
        <div class="results-grid">
            <div class="stat-card featured">
                <div class="stat-icon">
                    <i class="fas fa-road"></i>
                </div>
                <span class="stat-number">${data.potholes_detected || 0}</span>
                <span class="stat-label">Potholes Identified</span>
            </div>
            
            <div class="stat-card">
                <div class="stat-icon">
                    <i class="fas fa-hard-hat"></i>
                </div>
                <span class="stat-number">${materialLiters}</span>
                <span class="stat-label">Liters Required</span>
            </div>
            
            <div class="stat-card">
                <div class="stat-icon">
                    <i class="fas fa-clock"></i>
                </div>
                <span class="stat-number">${totalHours}</span>
                <span class="stat-label">Repair Hours</span>
            </div>
            
            ${videoInfoHtml}
        </div>
        
        <div class="cost-breakdown">
            <h3>
                <i class="fas fa-file-invoice-dollar"></i>
                Detailed Cost Breakdown
            </h3>
            <div class="cost-items">
                <div class="cost-item">
                    <span class="cost-label">
                        <i class="fas fa-hard-hat"></i>
                        Material Cost
                    </span>
                    <span class="cost-amount">₹${materialCost}</span>
                </div>
                <div class="cost-item">
                    <span class="cost-label">
                        <i class="fas fa-users"></i>
                        Labor Cost
                    </span>
                    <span class="cost-amount">₹${laborCost}</span>
                </div>
                <div class="cost-item">
                    <span class="cost-label">
                        <i class="fas fa-tools"></i>
                        Equipment & Transport
                    </span>
                    <span class="cost-amount">₹${equipmentTransport}</span>
                </div>
                <div class="cost-item">
                    <span class="cost-label">
                        <i class="fas fa-chart-line"></i>
                        Overhead (${overheadPercentage}%)
                    </span>
                    <span class="cost-amount">₹${overheadCost}</span>
                </div>
                <div class="cost-item total">
                    <span class="cost-label">
                        <i class="fas fa-receipt"></i>
                        Total Repair Cost
                    </span>
                    <span class="cost-amount">₹${totalCost}</span>
                </div>
            </div>
        </div>
        
        ${resultImageHtml}
        
        <div class="section-card">
            <div class="card-header">
                <i class="fas fa-download card-icon"></i>
                <h2>Export Report</h2>
                <p>Download detailed analysis report for your records</p>
            </div>
            <div style="text-align: center;">
                <button class="analyze-button" style="max-width: 300px;" onclick="exportReport(event)">
                    <i class="fas fa-file-pdf"></i>
                    Download PDF Report
                </button>
            </div>
        </div>
    `;
    
    resultsDiv.classList.remove('hidden');
    
    // Store data for PDF export
    currentResultsData = data;
    
    // Debug: Check if image loads
    if (data.result_image) {
        console.log('🖼️ Result image URL:', data.result_image);
        const img = new Image();
        img.onload = function() {
            console.log('✅ Image loaded successfully');
        };
        img.onerror = function() {
            console.log('❌ Image failed to load:', data.result_image);
        };
        img.src = data.result_image;
    }
}

function displayError(message) {
    // Clear progress simulation
    if (window.progressInterval) {
        clearInterval(window.progressInterval);
    }
    
    const errorDiv = document.getElementById('error');
    errorDiv.innerHTML = `
        <div class="error-icon">
            <i class="fas fa-exclamation-triangle"></i>
        </div>
        <h3>Analysis Failed</h3>
        <p>${message}</p>
        <button class="retry-button" onclick="clearError()">
            <i class="fas fa-redo"></i>
            Try Again
        </button>
    `;
    errorDiv.classList.remove('hidden');
}

function clearError() {
    document.getElementById('error').classList.add('hidden');
}

function exportReport(event) {
    // Prevent default if it's an event
    if (event) {
        event.preventDefault();
    }
    
    if (!currentResultsData) {
        showNotification('No analysis results available to export.', 'error');
        return;
    }
    
    // Get the button element properly
    const button = event?.target || document.querySelector('.analyze-button[onclick*="exportReport"]');
    if (!button) {
        showNotification('Could not find the export button.', 'error');
        return;
    }
    
    // Show loading state
    const originalHTML = button.innerHTML;
    button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating PDF...';
    button.disabled = true;
    
    console.log('Starting PDF generation with data:', currentResultsData);
    
    // Send request to generate PDF
    fetch('/generate_report', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(currentResultsData)
    })
    .then(response => {
        console.log('Response status:', response.status);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.blob();
    })
    .then(blob => {
        console.log('PDF blob received, size:', blob.size);
        
        if (blob.size === 0) {
            throw new Error('Empty PDF received');
        }
        
        // Create download link
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        
        // Create filename with timestamp
        const timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, '-');
        a.download = `pothole_report_${timestamp}.pdf`;
        
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        
        showNotification('PDF report downloaded successfully!', 'success');
    })
    .catch(error => {
        console.error('Error generating PDF:', error);
        showNotification(`Failed to generate PDF: ${error.message}`, 'error');
    })
    .finally(() => {
        // Always restore button state, even if there's an error
        console.log('Restoring button state');
        button.innerHTML = originalHTML;
        button.disabled = false;
    });
}

// Notification system
function showNotification(message, type = 'info') {
    // Remove existing notifications
    const existingNotifications = document.querySelectorAll('.notification');
    existingNotifications.forEach(notification => notification.remove());
    
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <div class="notification-content">
            <i class="fas fa-${getNotificationIcon(type)}"></i>
            <span>${message}</span>
        </div>
    `;
    
    // Add styles if not already added
    if (!document.querySelector('#notification-styles')) {
        const styles = document.createElement('style');
        styles.id = 'notification-styles';
        styles.textContent = `
            .notification {
                position: fixed;
                top: 20px;
                right: 20px;
                padding: 1rem 1.5rem;
                border-radius: 8px;
                color: white;
                z-index: 10000;
                animation: slideIn 0.3s ease;
                max-width: 400px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            .notification-success { background: var(--success); }
            .notification-error { background: var(--error); }
            .notification-info { background: var(--primary); }
            .notification-content {
                display: flex;
                align-items: center;
                gap: 0.75rem;
            }
            @keyframes slideIn {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
        `;
        document.head.appendChild(styles);
    }
    
    document.body.appendChild(notification);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.style.animation = 'slideOut 0.3s ease';
            setTimeout(() => notification.remove(), 300);
        }
    }, 5000);
}

function getNotificationIcon(type) {
    const icons = {
        'success': 'check-circle',
        'error': 'exclamation-triangle',
        'info': 'info-circle'
    };
    return icons[type] || 'info-circle';
}