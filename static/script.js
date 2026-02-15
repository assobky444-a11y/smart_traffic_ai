// Advanced Vehicle Tracking System - JavaScript

let currentVideoFile = null;
let currentVideoPath = null;
let currentJobId = null;
let processingStartTime = null;
let progressInterval = null;
let selectedConfidence = 0.25; // Global confidence value (default)

// ROI Variables
let roiPolygons = []; // saved polygons: {id, points}
let activeDrawingPoints = []; // points for current drawing
let roiCanvas = null;
let roiCtx = null;
let roiEnabled = false;
let drawingMode = false; // true when user is actively drawing a new ROI
let nextRoiId = 1; // simple incremental id for saved ROIs

// Smart Model Selection Mapping
const modelRecommendations = {
    'fast-low-high-speed': { model: 'yolo12n.pt', tip: '⚡ Fast highway tracking - Recommended!' },
    'fast-low-high-balanced': { model: 'yolo12n.pt', tip: '⚡ Optimized for highway speeds' },
    'fast-low-high-accuracy': { model: 'yolo12m.pt', tip: '⚖️ Balanced accuracy for highways' },
    
    'fast-medium-high-speed': { model: 'yolo12n.pt', tip: '⚡ Ultra-fast for busy highways' },
    'fast-medium-high-balanced': { model: 'yolo12m.pt', tip: '⚖️ Good balance for medium traffic' },
    'fast-medium-high-accuracy': { model: 'yolo26m.pt', tip: '🌆 Urban highway tracking' },
    
    'fast-high-high-speed': { model: 'yolo12n.pt', tip: '⚡ Maximum speed for dense traffic' },
    'fast-high-high-balanced': { model: 'yolo12m.pt', tip: '⚖️ Balanced dense traffic tracking' },
    'fast-high-high-accuracy': { model: 'best.pt', tip: '🎯 Best accuracy for crowded highways' },
    
    'medium-low-medium-speed': { model: 'yolo12s.pt', tip: '🔍 Good precision for urban roads' },
    'medium-low-medium-balanced': { model: 'yolo12m.pt', tip: '⚖️ Balanced for normal streets' },
    'medium-low-medium-accuracy': { model: 'yolo26m.pt', tip: '🌆 Best for urban street tracking' },
    
    'medium-medium-medium-speed': { model: 'yolo12m.pt', tip: '⚖️ Recommended for streets' },
    'medium-medium-medium-balanced': { model: 'yolo12m.pt', tip: '⚖️ Balanced for medium traffic' },
    'medium-medium-medium-accuracy': { model: 'best.pt', tip: '🎯 High accuracy on streets' },
    
    'medium-high-medium-speed': { model: 'yolo12m.pt', tip: '⚖️ Handles crowded streets well' },
    'medium-high-medium-balanced': { model: 'yolo26m.pt', tip: '🌆 Urban area specialization' },
    'medium-high-medium-accuracy': { model: 'best.pt', tip: '🎯 Best for congested areas' },
    
    'slow-low-low-speed': { model: 'yolo12s.pt', tip: '🔍 Precise tracking for slow traffic' },
    'slow-low-low-balanced': { model: 'yolo12m.pt', tip: '⚖️ Balanced slow traffic mode' },
    'slow-low-low-accuracy': { model: 'best.pt', tip: '🎯 Maximum precision' },
    
    'slow-medium-low-speed': { model: 'yolo12s.pt', tip: '🔍 Recommended for congestion' },
    'slow-medium-low-balanced': { model: 'yolo26m.pt', tip: '🌆 Urban congestion tracking' },
    'slow-medium-low-accuracy': { model: 'best.pt', tip: '🎯 Best accuracy for congestion' },
    
    'slow-high-low-speed': { model: 'yolo12s.pt', tip: '🔍 Handles heavy traffic well' },
    'slow-high-low-balanced': { model: 'best.pt', tip: '🎯 Recommended for gridlock' },
    'slow-high-low-accuracy': { model: 'best.pt', tip: '🎯 Maximum accuracy required' },
};

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const videoInput = document.getElementById('videoInput');
const uploadProgress = document.getElementById('uploadProgress');
const uploadProgressBar = document.getElementById('uploadProgressBar');
const uploadProgressText = document.getElementById('uploadProgressText');
const videoPreview = document.getElementById('videoPreview');
const previewVideo = document.getElementById('previewVideo');
const videoInfo = document.getElementById('videoInfo');
const configSection = document.getElementById('configSection');
const modelSelect = document.getElementById('modelSelect');
const confidenceSlider = document.getElementById('confidenceSlider');
const confidenceValue = document.getElementById('confidenceValue');
const iouSlider = document.getElementById('iouSlider');
const iouValue = document.getElementById('iouValue');
const maxAgeInput = document.getElementById('maxAgeInput');
const minHitsInput = document.getElementById('minHitsInput');
const startProcessingBtn = document.getElementById('startProcessingBtn');
const resetBtn = document.getElementById('resetBtn');
const processingSection = document.getElementById('processingSection');
const resultsSection = document.getElementById('resultsSection');
const loadingOverlay = document.getElementById('loadingOverlay');

// Initialize
window.isReviewModalOpen = false;
window.forceQualityUpdate = false;

document.addEventListener('DOMContentLoaded', () => {
    console.log('Page load: initializing UI and registering listeners');
    loadModels();
    loadHistory();
    setupEventListeners();

    // Ensure the "Show Low-Confidence Tracks" button always has a handler (defensive)
    try {
        const showBtn = document.getElementById('showLowConfBtn');
        if (showBtn && !showBtn._hasLowConfHandler) {
            showBtn.addEventListener('click', () => fetchLowConfTracks(window.currentJobId));
            showBtn._hasLowConfHandler = true;
        }
    } catch (e) { console.warn('Could not attach persistent low-conf handler', e); }

    // Apply initial recommendations and thresholds based on checked radios
    try { updateModelRecommendation(); } catch (e) { console.warn('updateModelRecommendation init failed', e); }
    try { updateConfidenceThreshold(); } catch (e) { console.warn('updateConfidenceThreshold init failed', e); }
});

// Update Model Recommendation Based on Questions - Progressive Selection
function updateModelRecommendation() {
    const speed = document.querySelector('input[name="vehicleSpeed"]:checked')?.value || '';
    const density = document.querySelector('input[name="trafficDensity"]:checked')?.value || '';
    const height = document.querySelector('input[name="cameraHeight"]:checked')?.value || '';
    const priority = document.querySelector('input[name="priority"]:checked')?.value || '';
    
    // Try to find exact match first
    const key = `${speed}-${density}-${height}-${priority}`;
    let recommendation = modelRecommendations[key];
    
    // If no exact match, find best match based on available selections
    if (!recommendation) {
        // Try partial matches in order of priority
        let partialKey = '';
        
        // Try 3-part key
        if (speed && density && height) {
            partialKey = `${speed}-${density}-${height}-${priority || 'balanced'}`;
            recommendation = modelRecommendations[partialKey];
        }
        
        // Try 2-part key
        if (!recommendation && speed && density) {
            const allKeys = Object.keys(modelRecommendations);
            const matches = allKeys.filter(k => k.startsWith(`${speed}-${density}`));
            if (matches.length > 0) {
                recommendation = modelRecommendations[matches[0]];
            }
        }
        
        // Try single part (speed only)
        if (!recommendation && speed) {
            const allKeys = Object.keys(modelRecommendations);
            const matches = allKeys.filter(k => k.startsWith(speed));
            if (matches.length > 0) {
                recommendation = modelRecommendations[matches[0]];
            }
        }
        
        // Default to yolo12n if still no match
        if (!recommendation) {
            recommendation = { model: 'yolo12n.pt', tip: '⚡ Default - Fast Model' };
        }
    }
    
    if (recommendation) {
        const modelTipEl = document.getElementById('modelTip');
        const availableModels = Array.from(modelSelect.options).map(o => o.value);
        if (modelTipEl) modelTipEl.style.display = 'inline';

        if (availableModels.includes(recommendation.model)) {
            modelSelect.value = recommendation.model;
            if (modelTipEl) modelTipEl.textContent = recommendation.tip;
        } else {
            const current = modelSelect.value || (availableModels[0] || 'yolo12n.pt');
            if (modelTipEl) modelTipEl.textContent = recommendation.tip + ' — Recommended model not available on server, using ' + current;
            console.warn('Recommended model', recommendation.model, 'not in available models; using', current);
        }

        // Log selected model to console
        console.log('🎯 Selected Model (applied or fallback):', modelSelect.value);
        console.log('📊 Model Configuration:', {
            vehicleSpeed: speed || '(pending)',
            trafficDensity: density || '(pending)',
            cameraHeight: height || '(pending)',
            priority: priority || '(pending)',
            recommendation: recommendation.tip
        });
    }
}

// Update Confidence Threshold Based on Vehicle Clarity
function updateConfidenceThreshold() {
    const clarity = document.querySelector('input[name="vehicleClarity"]:checked')?.value || '';
    
    const confidenceMap = {
        'clear': 0.70,      // Very Clear
        'moderate': 0.45,   // Moderate
        'unclear': 0.25     // Unclear
    };
    
    if (clarity && confidenceMap[clarity]) {
        selectedConfidence = confidenceMap[clarity];
        
        // Log to console
        console.log('🎯 Confidence Threshold Updated:', {
            clarity: clarity,
            confidenceValue: selectedConfidence,
            description: {
                'clear': 'Very Clear - High Detection Confidence',
                'moderate': 'Moderate - Balanced Detection',
                'unclear': 'Unclear - More Tolerant Detection'
            }[clarity]
        });
    }
}

// Setup Event Listeners
function setupEventListeners() {
    // Drag and drop
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    uploadArea.addEventListener('click', () => videoInput.click());
    
    // File input
    videoInput.addEventListener('change', handleFileSelect);
    
    // Buttons
    startProcessingBtn.addEventListener('click', startProcessing);
    resetBtn.addEventListener('click', resetForm);
    document.getElementById('newProcessBtn')?.addEventListener('click', resetForm);
    document.getElementById('viewHistoryBtn')?.addEventListener('click', () => {
        document.getElementById('historySection').scrollIntoView({ behavior: 'smooth' });
    });
    
    // Setup ROI handlers
    setupROIHandlers();
    
    // Initialize with optimal default values (for elements that still exist)
    if (iouSlider) iouSlider.value = 0.5;
    if (maxAgeInput) maxAgeInput.value = 40;
    if (minHitsInput) minHitsInput.value = 3;
}

// Drag and Drop Handlers
function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

// Handle File Upload
async function handleFile(file) {
    // Validate file type
    const validTypes = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska', 'video/x-msvideo'];
    const validExtensions = ['.mp4', '.avi', '.mov', '.mkv'];
    const fileExtension = file.name.toLowerCase().substring(file.name.lastIndexOf('.'));
    
    if (!validTypes.includes(file.type) && !validExtensions.includes(fileExtension)) {
        alert('Invalid file type. Please select a valid video file.');
        return;
    }
    
    // Validate file size (500MB max)
    const maxSize = 500 * 1024 * 1024;
    if (file.size > maxSize) {
        alert('File size is too large. Maximum size is 500 MB.');
        return;
    }
    
    currentVideoFile = file;
    
    // Show upload progress
    uploadProgress.style.display = 'block';
    uploadProgressBar.style.width = '0%';
    
    // Upload file
    const formData = new FormData();
    formData.append('video', file);
    
    try {
        const xhr = new XMLHttpRequest();
        
        // Progress tracking
        xhr.upload.addEventListener('progress', (e) => {
            if (e.lengthComputable) {
                const percentComplete = (e.loaded / e.total) * 100;
                uploadProgressBar.style.width = percentComplete + '%';
                uploadProgressText.textContent = `Uploading... ${Math.round(percentComplete)}%`;
            }
        });
        
        // Success handler
        xhr.addEventListener('load', () => {
            if (xhr.status === 200) {
                const response = JSON.parse(xhr.responseText);
                currentVideoPath = response.filepath;
                
                // Show video preview
                showVideoPreview(file);
                
                // Show config section
                configSection.style.display = 'block';
                // Apply recommendations/thresholds again now that config is visible
                try { updateModelRecommendation(); } catch (e) { console.warn('updateModelRecommendation on upload failed', e); }
                try { updateConfidenceThreshold(); } catch (e) { console.warn('updateConfidenceThreshold on upload failed', e); }
                configSection.scrollIntoView({ behavior: 'smooth' });
                
                uploadProgressText.textContent = 'Upload successful!';
            } else {
                alert('Video upload failed. Please try again.');
            }
        });
        
        // Error handler
        xhr.addEventListener('error', () => {
            alert('Error occurred during video upload.');
        });
        
        xhr.open('POST', '/api/upload');
        xhr.send(formData);
        
    } catch (error) {
        console.error('Upload error:', error);
        alert('Video upload failed: ' + error.message);
    }
}

// Show Video Preview
function showVideoPreview(file) {
    const url = URL.createObjectURL(file);
    previewVideo.src = url;
    videoPreview.style.display = 'block';
    
    previewVideo.onloadedmetadata = () => {
        const duration = previewVideo.duration;
        
        // حساب الساعات والدقائق والثواني
        const hours = Math.floor(duration / 3600);
        const minutes = Math.floor((duration % 3600) / 60);
        const seconds = Math.floor(duration % 60);
        
        // تنسيق الوقت ليعرض الساعات والدقائق والثواني
        const formattedTime = `${hours}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        
        // تغيير التنسيق ليكون المحتوى بجانب بعض وفي المنتصف
        videoInfo.innerHTML = `
            <div style="display: flex; gap: 15px; justify-content: center; flex-wrap: wrap; text-align: center;">
                <span><strong>Filename:</strong> ${file.name}</span>
                <span><strong>Size:</strong> ${(file.size / (1024 * 1024)).toFixed(2)} MB</span>
                <span><strong>Duration:</strong> ${formattedTime}</span>
            </div>
        `;
        
        // عرض section الـ ROI بعد اختيار الفيديو
        document.getElementById('roiSection').style.display = 'block';
    };
}



// Load Available Models
async function loadModels() {
    try {
        const response = await fetch('/api/models');
        const data = await response.json();
        
        console.log('Available models:', data.models);
        
        modelSelect.innerHTML = '';
        
        if (!data.models || data.models.length === 0) {
            console.warn('No models available, using default');
            const option = document.createElement('option');
            option.value = 'yolo12n.pt';
            option.textContent = 'yolo12n.pt (Default)';
            modelSelect.appendChild(option);
            modelSelect.value = 'yolo12n.pt';
            return;
        }
        
        data.models.forEach(model => {
            const option = document.createElement('option');
            // Ensure model name is just the filename without path
            const modelName = typeof model === 'string' ? model : model.name;
            const modelSize = typeof model === 'string' ? 0 : (model.size / (1024 * 1024));
            
            option.value = modelName;
            option.textContent = modelSize > 0 ? 
                `${modelName} (${modelSize.toFixed(2)} MB)` : 
                modelName;
            modelSelect.appendChild(option);
        });
        
        // Set default to yolo12n if available, otherwise first model
        const defaultModel = data.models.find(m => {
            const name = typeof m === 'string' ? m : m.name;
            return name === 'yolo12n.pt';
        });
        
        if (defaultModel) {
            modelSelect.value = typeof defaultModel === 'string' ? defaultModel : defaultModel.name;
        } else if (data.models.length > 0) {
            modelSelect.value = typeof data.models[0] === 'string' ? data.models[0] : data.models[0].name;
        }
        
        // Apply recommendation based on current question defaults (if function exists)
        try {
            updateModelRecommendation();
        } catch (e) {
            console.warn('updateModelRecommendation not available or failed:', e);
        }

        console.log('Selected default model:', modelSelect.value);
    } catch (error) {
        console.error('Error loading models:', error);
        // Fallback to default model
        modelSelect.innerHTML = '';
        const option = document.createElement('option');
        option.value = 'yolo12n.pt';
        option.textContent = 'yolo12n.pt (Default)';
        modelSelect.appendChild(option);
        modelSelect.value = 'yolo12n.pt';
    }
}

// Start Processing
async function startProcessing() {
    if (!currentVideoPath) {
        alert('Please upload a video file first.');
        return;
    }
    
    // Get model name - ensure it's just the filename
    let modelName = modelSelect.value.trim();
    // Remove 'modal/' prefix if it exists
    if (modelName.includes('/')) {
        modelName = modelName.split('/').pop();
    }
    
    // Validate model name
    if (!modelName || modelName === 'modal') {
        alert('Please select a valid model');
        return;
    }
    
    // Debug: Print ROI status
    console.log('=== ROI DEBUG ===');
    console.log('roiEnabled:', roiEnabled);
    console.log('saved ROIs count:', roiPolygons.length);
    console.log('active drawing points:', activeDrawingPoints);
    console.log('active drawing point count:', activeDrawingPoints ? activeDrawingPoints.length : 0);
    console.log('================');
    
    // Get configuration with optimized default values for accuracy
    const config = {
        video_path: currentVideoPath,
        model: modelName,
        confidence: selectedConfidence, // Updated by vehicle clarity question
        confidence_threshold: selectedConfidence, // used to split confident/low-confidence tracks
        iou_threshold: 0.5,  // Optimal for accurate vehicle matching
        max_age: 40,         // Keep tracks visible for longer periods
        min_hits: 3          // Confirm tracks after 3 frames for reliability
    };
    
    // Add ROI polygons if enabled and at least one saved polygon exists
    if (roiEnabled && Array.isArray(roiPolygons) && roiPolygons.length > 0) {
        // Send array of polygons where each polygon is an array of {x,y}
        config.roi_points = roiPolygons.map(p => p.points);
        // Include canvas and video sizes so backend can scale ROI to original video resolution
        const roiCanvas = window.roiCanvasElement;
        const roiVideo = document.getElementById('roiVideo');
        if (roiCanvas) {
            config.roi_canvas = { width: roiCanvas.width, height: roiCanvas.height };
        }
        if (roiVideo && roiVideo.videoWidth && roiVideo.videoHeight) {
            config.roi_video = { width: roiVideo.videoWidth, height: roiVideo.videoHeight };
        }
        console.log('✅ ROI filtering enabled (multiple polygons):', config.roi_points);
    } else if (roiEnabled && drawingMode && activeDrawingPoints.length > 0) {
        alert('Please save or clear your current ROI drawing before processing');
        return;
    } else if (roiEnabled && (!Array.isArray(roiPolygons) || roiPolygons.length === 0)) {
        alert('Please add at least one ROI polygon or disable region filtering');
        return;
    }
    
    // Read chosen accuracy and map to process_every_n_frames (low=3, medium=2, high=1)
    const accSel = document.querySelector('input[name="accuracy"]:checked');
    let processEvery = 2; // default = medium
    if (accSel && accSel.value) {
        processEvery = parseInt(accSel.value, 10) || 2;
    }
    config.process_every_n_frames = processEvery;

    // Log configuration to console
    console.log('🎯 Processing Configuration:', config, ' (process_every_n_frames=' + processEvery + ')');
    
    // Hide config section
    configSection.style.display = 'none';
    
    // Show processing section
    processingSection.style.display = 'block';
    processingSection.scrollIntoView({ behavior: 'smooth' });
    
    processingStartTime = Date.now();
    
    try {
        // Start processing
        const response = await fetch('/api/process', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(config)
        });
        
        const data = await response.json();
        
        console.log('Start processing response:', data);
        
        if (data.success) {
            currentJobId = data.job_id;
            console.log('Job created:', currentJobId);
            
            // Start polling for progress
            progressInterval = setInterval(checkProgress, 1000);
        } else {
            alert('Failed to start processing: ' + data.error);
        }
    } catch (error) {
        console.error('Processing error:', error);
        alert('Error occurred while starting processing: ' + error.message);
    }
}

// Check Processing Progress
async function checkProgress() {
    if (!currentJobId) return;
    
    try {
        const response = await fetch(`/api/job/${currentJobId}`);
        
        if (!response.ok) {
            console.error(`Job ${currentJobId} not found (${response.status})`);
            const errorData = await response.json();
            console.error('Error details:', errorData);
            
            // If job not found after 5 seconds, show error
            const elapsed = (Date.now() - processingStartTime) / 1000;
            if (elapsed > 5) {
                clearInterval(progressInterval);
                alert('Failed to track processing status. Job ID: ' + currentJobId);
                resetForm();
            }
            return;
        }
        
        const data = await response.json();
        console.log('Job status:', data);
        
        // Update progress
        const progress = data.progress || 0;
        const processedFrames = data.processed_frames || 0;
        const totalFrames = data.total_frames || 0;
        
        document.getElementById('processedFrames').textContent = processedFrames;
        document.getElementById('totalFrames').textContent = totalFrames;
        document.getElementById('progressPercent').textContent = progress + '%';
        document.getElementById('processingProgressBar').style.width = progress + '%';
        
        // Update elapsed time
        const elapsed = Math.floor((Date.now() - processingStartTime) / 1000);
        const minutes = Math.floor(elapsed / 60);
        const seconds = elapsed % 60;
        document.getElementById('elapsedTime').textContent = 
            `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        
        // Update status text
        if (data.status === 'processing') {
            document.getElementById('processingProgressText').textContent = 
                `Processing... ${processedFrames} / ${totalFrames} frames`;
        } else if (data.status === 'completed') {
            clearInterval(progressInterval);
            showResults(data.result, currentJobId);
        } else if (data.status === 'error') {
            clearInterval(progressInterval);
            alert('Error occurred during processing: ' + data.error);
            resetForm();
        }
    } catch (error) {
        console.error('Progress check error:', error);
    }
}

// Global helper: update the quality counts in the UI from a metrics object
function updateQualityUIFromMetrics(metrics, force=false) {
    try {
        // Block metric updates while review modal is open unless explicitly forced (e.g., after Save/Apply Correction)
        if (window.isReviewModalOpen && !force && !window.forceQualityUpdate) {
            console.log('Blocked UI metric update while review modal is open');
            return;
        }

        const confident = metrics.confident_count ?? metrics.high_conf_count ?? 0;
        // Fallback: if backend only provides low_conf_track_ids, derive low count from it
        const low = metrics.low_conf_count ?? (Array.isArray(metrics.low_conf_track_ids) ? metrics.low_conf_track_ids.length : 0);
        // Compute totals: prefer server-provided total_tracks, otherwise derive from counts
        const border = metrics.borderline_count ?? 0;
        const total = metrics.total_tracks ?? metrics.quality_tracks ?? (confident + low + border);

        // Compute percentages from counts when possible (ensures consistency after corrections)
        const confidentPct = (total && total > 0) ? (Number(confident) / Number(total) * 100) : (metrics.confident_pct !== undefined ? (Number(metrics.confident_pct) * (metrics.confident_pct <= 1 ? 100 : 1)) : 0);
        const lowPct = (total && total > 0) ? (Number(low) / Number(total) * 100) : (metrics.low_conf_pct !== undefined ? (Number(metrics.low_conf_pct) * (metrics.low_conf_pct <= 1 ? 100 : 1)) : 0);

        // Debug log to assist diagnosis when percentages seem stale
        console.log('updateQualityUIFromMetrics -> counts:', { confident, border, low, total }, 'computedPercents:', { confidentPct, lowPct });

        const setText = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = (typeof val === 'number' && id.endsWith('Pct')) ? (Number(val).toFixed(1) + '%') : val; };

        const elConf = document.getElementById('confidentTracks'); if (elConf) elConf.textContent = confident;
        const elConfPct = document.getElementById('confidentPct'); if (elConfPct) elConfPct.textContent = (typeof confidentPct === 'number') ? (Number(confidentPct).toFixed(1) + '%') : confidentPct;
        const elLow = document.getElementById('lowConfidenceTracks'); if (elLow) elLow.textContent = low;
        const elLowPct = document.getElementById('lowConfidencePct'); if (elLowPct) elLowPct.textContent = (typeof lowPct === 'number') ? (Number(lowPct).toFixed(1) + '%') : lowPct;

        const showBtn = document.getElementById('showLowConfBtn'); if (showBtn) showBtn.style.display = (Number(low) > 0 || (Array.isArray(window._lowConfTracks) && window._lowConfTracks.length > 0)) ? 'inline-flex' : 'none';
    } catch (e) {
        console.warn('updateQualityUIFromMetrics failed', e);
    }
}

// Show Results
async function showResults(result, jobId, noScroll=false) {
    // Hide processing section
    processingSection.style.display = 'none';
    
    // Check if result exists
    if (!result) {
        console.error('No result data received');
        resultsSection.style.display = 'none';
        processingSection.style.display = 'block';
        processingSection.innerHTML = '<div class="error-message">Error: No result data received</div>';
        return;
    }
    
    // Show results section
    resultsSection.style.display = 'block';
    // Only scroll into view when not explicitly suppressed (prevents unwanted UI jumps)
    if (!noScroll && typeof resultsSection.scrollIntoView === 'function') {
        try { resultsSection.scrollIntoView({ behavior: 'smooth' }); } catch (e) { /* ignore */ }
    }
    
    // Store analysis folder globally for request analysis button
    console.log('Result object:', result);
    console.log('CSV path from result:', result.csv_path);
    
    // Extract folder name from csv_path
    const csvPath = result.csv_path.replace(/\\/g, '/');
    let folderPath = '';
    if (csvPath.includes('/unified_output/')) {
        folderPath = csvPath.split('/unified_output/')[1].split('/tracks.csv')[0];
    } else if (csvPath.includes('unified_output/')) {
        folderPath = csvPath.split('unified_output/')[1].split('/tracks.csv')[0];
    } else {
        const parts = csvPath.split('/');
        folderPath = parts[parts.length - 2];
    }
    
    window.currentFolderName = folderPath;
    window.currentJobId = jobId; // store job id for UI actions
    console.log('Stored window.currentFolderName:', window.currentFolderName);
    console.log('Stored window.currentJobId:', window.currentJobId);
    
    // Update results
    const meta = result.meta;
    // Debug: log authoritative quality metrics returned from the backend for inspection
    console.log('quality_metrics (from meta):', meta ? meta.quality_metrics : undefined);
    
    // Get data from quality_metrics or fallback to top-level
    const totalTracks = meta.quality_metrics?.total_tracks || meta.num_tracks || meta.total_tracks || 0;
    const totalDetections = meta.processed_frames_with_detection || meta.total_detections || 0;
    const qualityScore = meta.quality_metrics?.quality_score || 0;

    // Get vehicle class distribution from backend API (needed to compute total vehicles)
    let classDistribution = {};
    try {
        const vehicleClassesResponse = await fetch(`/api/job/${jobId}/vehicle_classes`);
        const classesData = await vehicleClassesResponse.json();
        console.log('vehicle_classes API response:', classesData);
        classDistribution = classesData.distribution || {};
    } catch (e) {
        console.error('Error fetching vehicle classes:', e);
    }

    // Helper: update quality metrics UI from a metrics object (used after corrections)
    function updateQualityUIFromMetrics(metrics) {
        try {
            const confident = metrics.confident_count ?? metrics.high_conf_count ?? 0;
            // Fallback: if backend only provides low_conf_track_ids, derive low count from it
            const low = metrics.low_conf_count ?? (Array.isArray(metrics.low_conf_track_ids) ? metrics.low_conf_track_ids.length : 0);
            const border = metrics.borderline_count ?? 0;

            // Prefer authoritative total if provided, otherwise derive from counts
            const total = metrics.total_tracks ?? metrics.quality_tracks ?? (confident + low + border);

            // Compute percentages from counts when possible (ensures consistency after corrections)
            const confidentPct = (total && total > 0) ? (Number(confident) / Number(total) * 100) : (metrics.confident_pct !== undefined ? (Number(metrics.confident_pct) * (metrics.confident_pct <= 1 ? 100 : 1)) : 0);
            const lowPct = (total && total > 0) ? (Number(low) / Number(total) * 100) : (metrics.low_conf_pct !== undefined ? (Number(metrics.low_conf_pct) * (metrics.low_conf_pct <= 1 ? 100 : 1)) : 0);

            console.log('updateQualityUIFromMetrics (inner) -> counts:', { confident, border, low, total }, 'computedPercents:', { confidentPct, lowPct });

            safeSetText('confidentTracks', confident);
            safeSetText('confidentPct', Number(confidentPct).toFixed(1) + '%');
            safeSetText('lowConfidenceTracks', low);
            safeSetText('lowConfidencePct', Number(lowPct).toFixed(1) + '%');

            // Show/hide low-conf review button based on low count
            const showBtn = document.getElementById('showLowConfBtn');
            if (showBtn) {
                // keep button visible if server reports low items OR we have loaded low-conf previews locally
                const shouldShow = (Number(low) > 0) || (Array.isArray(window._lowConfTracks) && window._lowConfTracks.length > 0);
                showBtn.style.display = shouldShow ? 'inline-flex' : 'none';
            }
        } catch (e) {
            console.warn('updateQualityUIFromMetrics failed', e);
        }
    }

    // Compute totals from backend meta and class distribution, prefer backend track total for consistency with high/low split
    const backendTotal = meta.quality_metrics?.total_tracks ?? meta.num_tracks ?? meta.total_tracks ?? 0;
    let classSum = 0;
    try {
        if (classDistribution && Object.keys(classDistribution).length > 0) {
            classSum = Object.values(classDistribution).reduce((acc, v) => acc + (Number(v) || 0), 0);
        }
    } catch (e) {
        console.warn('Error computing class distribution sum', e);
    }

    // Decide which total to display: prefer backendTotal if present, otherwise classSum, otherwise fallback totalTracks
    let totalVehiclesCount = backendTotal || (classSum > 0 ? classSum : totalTracks);

    // If both present and mismatch, log details and prefer backendTotal
    if (classSum > 0 && backendTotal > 0 && classSum !== backendTotal) {
        console.warn(`Total mismatch (job=${jobId}, folder=${folderPath}): backend total=${backendTotal}, class sum=${classSum}. Showing backend total (tracks) for consistency.`);
    }

    const totalTracksEl = document.getElementById('totalTracks');
    if (totalTracksEl) totalTracksEl.textContent = totalVehiclesCount;
    const totalDetectionsEl = document.getElementById('totalDetections');
    if (totalDetectionsEl) totalDetectionsEl.textContent = totalDetections;

    // Indicate source used for Total Vehicles and show both numbers when inconsistent (and show UI note)
    const sourceElem = document.getElementById('totalVehiclesSource');
    const mismatchNote = document.getElementById('mismatchNote');
    if (sourceElem) {
        if (classSum > 0 && backendTotal > 0 && classSum !== backendTotal) {
            sourceElem.textContent = `(source: tracks=${backendTotal}, classes=${classSum})`;
            sourceElem.style.color = '#f39c12';
            sourceElem.style.fontWeight = '600';
            if (mismatchNote) {
                mismatchNote.style.display = 'block';
                mismatchNote.textContent = `⚠️ Discrepancy detected: class counts (${classSum}) differ from tracks (${backendTotal}). Showing tracks (exported IDs) as authoritative.`;
            }
        } else if (classSum > 0) {
            sourceElem.textContent = '(source: class distribution)';
            sourceElem.style.color = '#ddd';
            sourceElem.style.fontWeight = 'normal';
            if (mismatchNote) { mismatchNote.style.display = 'none'; mismatchNote.textContent = ''; }
        } else {
            sourceElem.textContent = '(source: tracks)';
            sourceElem.style.color = '#ddd';
            sourceElem.style.fontWeight = 'normal';
            if (mismatchNote) { mismatchNote.style.display = 'none'; mismatchNote.textContent = ''; }
        }
    }

    // Show unclassified count if present in meta
    const unclassifiedVal = meta.unclassified_tracks ?? 0;
    const unclassifiedBadge = document.getElementById('unclassifiedBadge');
    if (unclassifiedBadge) {
        const countEl = document.getElementById('unclassifiedCount');
        if (Number(unclassifiedVal) > 0) {
            unclassifiedBadge.style.display = 'block';
            if (countEl) countEl.textContent = Number(unclassifiedVal);
        } else {
            unclassifiedBadge.style.display = 'none';
            if (countEl) countEl.textContent = '0';
        }
    }

    // Update quality/accuracy if available
    const qualityElem = document.getElementById('accuracy');
    if (qualityElem) qualityElem.textContent = qualityScore.toFixed(1) + '%';

    // Read high/low confidence split from meta (if present)
    const highCount = meta.quality_metrics?.high_conf_count;
    const highPct = meta.quality_metrics?.high_conf_pct;
    const lowCount = meta.quality_metrics?.low_conf_count;
    const lowPct = meta.quality_metrics?.low_conf_pct;
    const confThresh = meta.quality_metrics?.confidence_threshold ?? meta.confidence_threshold ?? null;

    const reliableFlag = meta.quality_metrics?.reliable;

    // Helper: safely set textContent when element exists
    function safeSetText(id, value) {
        const el = document.getElementById(id);
        if (el) el.textContent = value;
    }

    function safeSetLabel(id, value) {
        const el = document.getElementById(id);
        if (el) el.textContent = value;
    }

    const qWarn = document.getElementById('qualityWarning');

    if (highCount !== undefined && lowCount !== undefined) {
        // Prefer authoritative total when available; otherwise derive from counts and optional borderline count
        const borderlineCount = meta.quality_metrics?.borderline_count ?? 0;
        const authoritativeTotal = backendTotal || (Number(highCount) + Number(lowCount) + Number(borderlineCount));

        // Compute percentages from counts to keep values consistent in the UI
        const calcHighPct = (authoritativeTotal && Number(authoritativeTotal) > 0) ? (Number(highCount) / Number(authoritativeTotal) * 100) : ((highPct !== undefined && highPct !== null) ? (Number(highPct) * 100) : 0);
        const calcLowPct = (authoritativeTotal && Number(authoritativeTotal) > 0) ? (Number(lowCount) / Number(authoritativeTotal) * 100) : ((lowPct !== undefined && lowPct !== null) ? (Number(lowPct) * 100) : 0);

        safeSetText('confidentTracks', highCount);
        safeSetText('confidentPct', Number(calcHighPct).toFixed(1) + '%');
        safeSetText('lowConfidenceTracks', lowCount);
        safeSetText('lowConfidencePct', Number(calcLowPct).toFixed(1) + '%');

        if (confThresh !== null) {
            const pct = Math.round(Number(confThresh) * 100);
            safeSetLabel('confidentThresholdLabel', `(≥ ${pct}%)`);
            safeSetLabel('lowThresholdLabel', `(< ${pct}%)`);
        }

        // Sanity check: high + low should equal backend total tracks (if available)
        const totalCheck = Number(highCount) + Number(lowCount);
        if (backendTotal && totalCheck !== Number(backendTotal)) {
            console.warn(`Confident/Low totals (${totalCheck}) do not match backend total tracks (${backendTotal}). classSum=${classSum} (job=${jobId}, folder=${folderPath})`);
            // Helpful hint: prefer backend total for authoritative track count
        } else if (!backendTotal && totalVehiclesCount && totalCheck !== Number(totalVehiclesCount)) {
            console.warn(`Confident/Low totals (${totalCheck}) do not match displayed total vehicles (${totalVehiclesCount}). classSum=${classSum} (job=${jobId}, folder=${folderPath})`);
        }

        // Reliability warning for small samples
        if (qWarn) {
            if (reliableFlag === false) {
                qWarn.style.display = 'block';
                qWarn.textContent = '⚠️ Warning: Metrics may be unreliable for small number of tracks (<5).';
            } else {
                qWarn.style.display = 'none';
                qWarn.textContent = '';
            }
        }

        // Show low-confidence review button if any low-confidence or borderline tracks exist
        try {
            const showBtn = document.getElementById('showLowConfBtn');
            if (showBtn) {
                if (Number(lowCount) > 0) {
                    showBtn.style.display = 'inline-flex';
                    showBtn.onclick = () => fetchLowConfTracks(window.currentJobId);
                    // show regenerate button nearby
                    const regenBtn = document.getElementById('regeneratePreviewsBtn');
                    if (regenBtn) {
                        regenBtn.style.display = 'inline-flex';
                        regenBtn.onclick = async function() {
                            const folder = window.currentFolderName;
                            if (!folder) return alert('Folder not selected');
                            this.disabled = true;
                            this.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Regenerating...';
                            try {
                                const resp = await fetch(`/api/regenerate_previews/${folder}`, {method: 'POST'});
                                const data = await resp.json();
                                alert('Regeneration complete: ' + JSON.stringify(data));
                            } catch (e) {
                                alert('Failed to regenerate');
                            }
                            this.disabled = false;
                            this.innerHTML = '<i class="fas fa-sync"></i> Regenerate Previews';
                        }
                    }
                } else {
                    showBtn.style.display = 'none';
                    const regenBtn = document.getElementById('regeneratePreviewsBtn');
                    if (regenBtn) regenBtn.style.display = 'none';
                }
            }
        } catch (e) {
            console.warn('Error setting up low-conf button', e);
        }
    } else {
        // If backend did not provide split, show placeholders
        safeSetText('confidentTracks', 'N/A');
        safeSetText('confidentPct', 'N/A');
        safeSetText('lowConfidenceTracks', 'N/A');
        safeSetText('lowConfidencePct', 'N/A');
        if (confThresh !== null) {
            const pct = Math.round(Number(confThresh) * 100);
            safeSetLabel('confidentThresholdLabel', `(≥ ${pct}%)`);
            safeSetLabel('lowThresholdLabel', `(< ${pct}%)`);
        }
        if (qWarn) {
            qWarn.style.display = 'none';
            qWarn.textContent = '';
        }
    }
    
    // Calculate processing speed
    const processingTime = (Date.now() - processingStartTime) / 1000;
    
    // `classDistribution` was fetched earlier to compute total vehicles.
    // Reuse that value here; ensure it's an object as a fallback.
    classDistribution = classDistribution || {};
    
    const vehicleClassCount = Object.keys(classDistribution).length;
    const vehicleClassElem = document.getElementById('vehicleClassCount');
    if (vehicleClassElem) vehicleClassElem.textContent = vehicleClassCount;
    
    // Display vehicle classes breakdown (only if element exists on the page)
    const breakdownDiv = document.getElementById('vehicleClassesBreakdown');
    if (breakdownDiv) breakdownDiv.innerHTML = '';
    
    const colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#00f2fe', '#43e97b', '#fa709a', '#fee140'];
    let colorIndex = 0;
    
    if (breakdownDiv) {
        Object.entries(classDistribution).sort((a, b) => b[1] - a[1]).forEach(([className, count]) => {
            const color = colors[colorIndex % colors.length];
            const card = document.createElement('div');
            card.style.cssText = `
                background: linear-gradient(135deg, ${color}, ${color}dd);
                padding: 20px;
                border-radius: 12px;
                color: white;
                text-align: center;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                transition: all 0.3s ease;
            `;
            card.innerHTML = `
                <div style="font-size: 2.5rem; font-weight: 700; margin-bottom: 5px;">${count}</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">${className}</div>
            `;
            card.addEventListener('mouseenter', () => {
                card.style.transform = 'translateY(-5px)';
                card.style.boxShadow = '0 10px 25px rgba(0,0,0,0.15)';
            });
            card.addEventListener('mouseleave', () => {
                card.style.transform = 'translateY(0)';
                card.style.boxShadow = '0 4px 15px rgba(0,0,0,0.1)';
            });
            breakdownDiv.appendChild(card);
            colorIndex++;
        });
        
        if (vehicleClassCount === 0) {
            breakdownDiv.innerHTML = '<div style="grid-column: 1/-1; text-align: center; color: #999; padding: 20px;">No vehicle classification data available</div>';
        }
    }
    
    // Setup download buttons (if they exist)
    // folderPath already extracted above
    const downloadCsvBtn = document.getElementById('downloadCsvBtn');
    const downloadMetaBtn = document.getElementById('downloadMetaBtn');
    
    if (downloadCsvBtn) {
        downloadCsvBtn.href = `/api/download/${folderPath}/tracks.csv`;
    }
    if (downloadMetaBtn) {
        downloadMetaBtn.href = `/api/download/${folderPath}/meta.json`;
    }
    
    // Add click-to-open for the low-confidence modal image
    try {
        const lowImg = document.getElementById('lowConfImage');
        if (lowImg) {
            lowImg.style.cursor = 'zoom-in';
            lowImg.addEventListener('click', () => {
                if (lowImg.src) window.open(lowImg.src, '_blank');
            });
        }
    } catch (e) {
        console.warn('Could not attach zoom handler to lowConfImage', e);
    }

    // Reload history
    loadHistory();
}

// Refresh results (non-intrusive update used after corrections) 🔧
async function refreshResults(jobId) {
    try {
        const resp = await fetch(`/api/job/${jobId}`, {cache: 'no-store'});
        if (!resp.ok) return;
        const job = await resp.json();
        const result = job.result;
        if (!result) return;
        const meta = result.meta || {};

        // Use showResults to perform a full authoritative UI refresh so counts and breakdown stay in sync
        // Suppress scrolling here to avoid jumping when performing a background refresh (e.g., after closing modal)
        try {
            await showResults(result, jobId, true);
            return;
        } catch (e) {
            console.warn('refreshResults: showResults failed, falling back to partial update', e);
        }

        // Update totals (fallback)
        const backendTotal = meta.quality_metrics?.total_tracks ?? meta.num_tracks ?? meta.total_tracks ?? 0;
        const totalDetections = meta.processed_frames_with_detection || meta.total_detections || 0;
        const totalTracksEl = document.getElementById('totalTracks'); if (totalTracksEl) totalTracksEl.textContent = backendTotal || (meta.num_tracks||meta.total_tracks||0);
        const totalDetectionsEl = document.getElementById('totalDetections'); if (totalDetectionsEl) totalDetectionsEl.textContent = totalDetections;

        // Unclassified badge
        const unclassifiedVal = meta.unclassified_tracks ?? 0;
        const unclassifiedBadge = document.getElementById('unclassifiedBadge');
        if (unclassifiedBadge) {
            const countEl = document.getElementById('unclassifiedCount');
            if (Number(unclassifiedVal) > 0) {
                unclassifiedBadge.style.display = 'block';
                if (countEl) countEl.textContent = Number(unclassifiedVal);
            } else {
                unclassifiedBadge.style.display = 'none';
                if (countEl) countEl.textContent = '0';
            }
        }

        // Update quality metrics forcibly (bypass modal blocking)
        if (meta.quality_metrics) {
        updateQualityUIFromMetrics(meta.quality_metrics, true);

        // Sanity check: ensure confident + borderline + low equals total_tracks
        try {
            const qm = meta.quality_metrics || {};
            const high = qm.confident_count ?? qm.high_conf_count ?? 0;
            const border = qm.borderline_count ?? 0;
            const low = qm.low_conf_count ?? (Array.isArray(qm.low_conf_track_ids) ? qm.low_conf_track_ids.length : 0);
            if (qm.total_tracks && (Number(high) + Number(border) + Number(low) !== Number(qm.total_tracks))) {
                const qw = document.getElementById('qualityWarning');
                if (qw) {
                    qw.style.display = 'block';
                    qw.textContent = `⚠️ Mismatch: confident(${high}) + borderline(${border}) + low(${low}) != total(${qm.total_tracks}). Values shown are authoritative.`;
                }
            }
        } catch (e) {
            console.warn('Quality metrics sanity check failed', e);
        }
    }

        // Refresh vehicle class distribution
        try {
            const vc = await fetch(`/api/job/${jobId}/vehicle_classes`);
            const vcData = await vc.json();
            const classDistribution = vcData.distribution || {};
            const vehicleClassElem = document.getElementById('vehicleClassCount');
            if (vehicleClassElem) vehicleClassElem.textContent = Object.keys(classDistribution).length;
            const breakdownDiv = document.getElementById('vehicleClassesBreakdown');
            if (breakdownDiv) {
                breakdownDiv.innerHTML = '';
                const colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#00f2fe', '#43e97b', '#fa709a', '#fee140'];
                let colorIndex = 0;
                Object.entries(classDistribution).sort((a, b) => b[1] - a[1]).forEach(([className, count]) => {
                    const color = colors[colorIndex % colors.length];
                    const card = document.createElement('div');
                    card.style.cssText = `background: linear-gradient(135deg, ${color}, ${color}dd); padding: 20px; border-radius: 12px; color: white; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.1); transition: all 0.3s ease;`;
                    card.innerHTML = `<div style="font-size: 2.5rem; font-weight: 700; margin-bottom: 5px;">${count}</div><div style="font-size: 0.9rem; opacity: 0.9;">${className}</div>`;
                    breakdownDiv.appendChild(card);
                    colorIndex++;
                });
                if (Object.keys(classDistribution).length === 0) {
                    breakdownDiv.innerHTML = '<div style="grid-column: 1/-1; text-align: center; color: #999; padding: 20px;">No vehicle classification data available</div>';
                }
            }
        } catch (e) {
            console.warn('refreshResults: failed to refresh vehicle classes', e);
        }

    } catch (e) {
        console.error('refreshResults failed', e);
    }
}

// Send tracking to admin for advanced analysis
async function sendToAdmin(analysisId, folderName) {
    if (!confirm('Do you want to send this tracking to admin for advanced analysis?')) return;
    
    try {
        const response = await fetch('/api/request_analysis_auto', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ folder_name: folderName })
        });
        
        const data = await response.json();
        
        if (data.success) {
            alert('✅ Request sent successfully!\nRequest ID: ' + data.request_id);
            // Redirect to my requests page
            window.location.href = '/my_requests';
        } else {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        console.error('Error sending request:', error);
        alert('Error occurred while sending the request');
    }
}

// Delete result function
async function deleteResult(folderName) {
    if (!confirm('Are you sure you want to delete this tracking result? This action cannot be undone.')) {
        return;
    }

    try {
        const response = await fetch(`/api/delete_result/${folderName}`, {
            method: 'DELETE',
            headers: {'Content-Type': 'application/json'}
        });

        const data = await response.json();

        if (data.success) {
            alert('✅ Tracking result deleted successfully');
            loadHistory(); // Reload the history
        } else {
            alert('Error: ' + (data.error || 'Failed to delete result'));
        }
    } catch (error) {
        console.error('Error deleting result:', error);
        alert('Error occurred while deleting the result');
    }
}

// View result function - find request_id and redirect
async function viewResult(folderName) {
    try {
        const response = await fetch(`/api/get_request_id_by_folder/${folderName}`);
        const data = await response.json();

        if (data.success) {
            // Redirect to request_analysis page with request_id
            window.location.href = `/request_analysis?request_id=${data.request_id}`;
        } else {
            alert('This tracking result has not been submitted as an analysis request yet.\n\nPlease use "Send For Analysis" button to submit it first.');
        }
    } catch (error) {
        console.error('Error viewing result:', error);
        alert('Error occurred while viewing the result');
    }
}

async function loadHistory() {
    try {
        const response = await fetch('/api/results');
        const data = await response.json();
        
        const tbody = document.getElementById('historyTableBody');
        tbody.innerHTML = '';
        
        if (data.results.length === 0) {
            tbody.innerHTML = '<tr><td colspan="4" style="text-align: center;">No previous results</td></tr>';
            return;
        }
        
        data.results.forEach(result => {
            const meta = result.meta;
            const row = document.createElement('tr');
            
            // Fix date - extract from folder name (tracking_YYYYMMDD_HHMMSS)
            let date = 'N/A';
            const folderName = result.folder;
            const dateMatch = folderName.match(/tracking_(\d{8})_(\d{6})/);
            if (dateMatch) {
                const dateStr = dateMatch[1]; // YYYYMMDD
                const timeStr = dateMatch[2]; // HHMMSS
                const year = dateStr.substring(0, 4);
                const month = dateStr.substring(4, 6);
                const day = dateStr.substring(6, 8);
                const hour = timeStr.substring(0, 2);
                const minute = timeStr.substring(2, 4);
                date = `${year}-${month}-${day} ${hour}:${minute}`;
            }
            
            // Fix filename extraction
            const filename = meta.video_path ? meta.video_path.split('\\').pop().split('/').pop() : 'Unknown';
            
            const frames = meta.total_frames !== undefined ? meta.total_frames : (meta.frame_count || 'N/A');

            row.innerHTML = `
                <td>${date}</td>
                <td>${filename}</td>
                <td>${frames}</td>
                <td>
                    <div style="display: flex; gap: 8px; align-items: center;">
                        <button onclick="sendForAnalysis('${result.folder}', this)" class="btn" style="background: white; color: #667eea; padding: 8px 16px; font-size: 0.85rem; font-weight: 600; border: none; border-radius: 8px; cursor: pointer; transition: all 0.3s ease; box-shadow: 0 4px 15px rgba(0,0,0,0.15);">
                            <i class="fas fa-paper-plane"></i> Send Analysis Request
                        </button>
                        <a href="/api/download/${result.folder}/tracks.csv" class="btn btn-primary" style="padding: 5px 10px; font-size: 0.85rem; display: inline-flex; align-items: center; justify-content: center;">
                            <i class="fas fa-download"></i> Download
                        </a>
                        <button onclick="deleteResult('${result.folder}')" class="btn btn-danger" style="padding: 5px 10px; font-size: 0.85rem; background: #dc3545;">
                            <i class="fas fa-trash"></i> Delete
                        </button>
                    </div>
                </td>
            `;
            
            tbody.appendChild(row);
        });
    } catch (error) {
        console.error('Error loading history:', error);
    }
}

// Send Analysis Request from history row
async function sendForAnalysis(folderName, btn) {
    if (!confirm('Do you want to send this tracking as an analysis request?')) return;

    const originalHTML = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Sending...';

    try {
        const response = await fetch('/api/request_analysis_auto', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ folder_name: folderName })
        });

        const data = await response.json();

        if (data.success) {
            // Redirect to request analysis page for this request
            window.location.href = `/request_analysis?request_id=${data.request_id}`;
        } else {
            alert('Error: ' + (data.error || 'Failed to send request'));
            btn.disabled = false;
            btn.innerHTML = originalHTML;
        }
    } catch (error) {
        console.error('Error sending analysis request:', error);
        alert('Error occurred while sending the request');
        btn.disabled = false;
        btn.innerHTML = originalHTML;
    }
}

// Global processRequest helper available for both admin and user flows
function processRequest(requestId, role = 'admin') {
    if (!requestId || requestId === 'None' || requestId === '') {
        alert('Request ID not available. Please send an analysis request first or open the request from My Requests.');
        return;
    }
    if (role === 'admin') {
        window.location.href = `/admin/line_drawing?request_id=${requestId}`;
    } else {
        window.location.href = `/user/line_drawing?request_id=${requestId}`;
    }
}

// Reset Form
function resetForm() {
    // Reset variables
    currentVideoFile = null;
    currentVideoPath = null;
    currentJobId = null;
    processingStartTime = null;
    
    if (progressInterval) {
        clearInterval(progressInterval);
        progressInterval = null;
    }
    
    // Reset UI
    videoInput.value = '';
    uploadProgress.style.display = 'none';
    videoPreview.style.display = 'none';
    configSection.style.display = 'none';
    processingSection.style.display = 'none';
    resultsSection.style.display = 'none';
    
    // Reset sliders (with null checks since they may be hidden)
    if (confidenceSlider) confidenceSlider.value = 0.25;
    if (confidenceValue) confidenceValue.textContent = '0.25';
    if (iouSlider) iouSlider.value = 0.45;
    if (iouValue) iouValue.textContent = '0.45';
    if (maxAgeInput) maxAgeInput.value = 30;
    if (minHitsInput) minHitsInput.value = 3;
    
    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Utility Functions
function showLoading() {
    loadingOverlay.style.display = 'flex';
}

function hideLoading() {
    loadingOverlay.style.display = 'none';
}

// Format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

// Format duration
function formatDuration(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    if (hours > 0) {
        return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    return `${minutes}:${secs.toString().padStart(2, '0')}`;
}
// ==================== Low-Confidence Batch Grid Review ====================

// Inline notice helper for the low-confidence modal (replaces blocking alerts)
function showLowConfNotice(msg, level = 'info', autoHide = true) {
    const el = document.getElementById('lowConfNotice');
    if (!el) return console.warn('lowConfNotice element not found:', msg);
    el.textContent = msg;
    el.style.display = 'block';
    if (level === 'error') el.style.color = '#ff6b6b';
    else if (level === 'success') el.style.color = '#7ee787';
    else el.style.color = '#ffcc00';

    if (autoHide) {
        clearTimeout(el._hideTimer);
        el._hideTimer = setTimeout(() => { el.style.display = 'none'; }, 4000);
    }
}

let _lowConfTracks = [];
let _lowConfByClass = {};
let _categories = [];
let _currentCategory = null;
let _batchSize = 9; // 3x3
let _batchIndexByCat = {};
let _unclassifiedList = [];
let _selectedSet = new Set(); // selected IDs in current batch

async function fetchLowConfTracks(jobId, openModal=true) {
    if (!jobId) return showLowConfNotice('Job id not available', 'error');
    try {
        showLoading();
        const resp = await fetch(`/api/job/${jobId}/low_confidence_tracks`, {cache: 'no-store'});
        const text = await resp.text();
        hideLoading();
        let data = null;
        try {
            data = JSON.parse(text);
        } catch (e) {
            console.error('Invalid JSON from low_conf endpoint:', text.slice(0,400));
            showLowConfNotice('Invalid server response while loading low-confidence tracks. See console for details.', 'error');
            return;
        }
        if (data.error) { showLowConfNotice('Error fetching low-confidence tracks: ' + data.error, 'error'); return; }

        const ids = Array.isArray(data.low_conf_track_ids) ? data.low_conf_track_ids : (Array.isArray(data.low_conf_ids) ? data.low_conf_ids : null);
        const returnedTracks = Array.isArray(data.tracks) ? data.tracks : [];
        const trackMap = {};
        returnedTracks.forEach(t => { try { trackMap[t.track_id] = t; } catch (e) {} });

        if (ids && ids.length >= 0) {
            _lowConfTracks = ids.map(id => trackMap[id] ? trackMap[id] : { track_id: id, image: '', n_points:0, median_conf:0, mean_conf:0, pct_above:0, current_class: 'unclassified' });
        } else {
            _lowConfTracks = returnedTracks.slice();
        }

        // organize by predicted class
        _lowConfByClass = {};
        _lowConfTracks.forEach(t => {
            const cls = (t.current_class || 'unclassified');
            if (!(_lowConfByClass[cls])) _lowConfByClass[cls] = [];
            _lowConfByClass[cls].push(t);
        });

        _categories = Object.keys(_lowConfByClass).filter(c => c && c !== 'unclassified');
        _categories.sort((a,b) => _lowConfByClass[b].length - _lowConfByClass[a].length);
        _categories.forEach(c => { if (_batchIndexByCat[c] === undefined) _batchIndexByCat[c] = 0; });

        _unclassifiedList = (_lowConfByClass['unclassified'] || []).slice();

        // update top-level unclassified badge
        try {
            const uc = document.getElementById('unclassifiedCount');
            const ub = document.getElementById('unclassifiedBadge');
            if (uc) uc.textContent = _unclassifiedList.length;
            if (ub) ub.style.display = (_unclassifiedList.length > 0) ? 'inline-block' : 'none';
        } catch (e) { console.warn('Could not update unclassified badge', e); }

            // Ensure Show Low-Confidence button reflects the authoritative low list we just loaded
            try {
                const showBtn = document.getElementById('showLowConfBtn');
                if (showBtn) showBtn.style.display = (_lowConfTracks && _lowConfTracks.length > 0) ? 'inline-flex' : 'none';
            } catch (e) { /* ignore */ }

        // If there are no low-conf tracks, clear the modal UI instead of showing an alert.
        if (_lowConfTracks.length === 0) {
            // clear local structures and render empty grids
            _lowConfByClass = {};
            _categories = [];
            _currentCategory = null;
            _selectedSet.clear();
            renderCategorySlider();
            renderExpectedBatch();
            await renderUnclassifiedGrid();

            // hide modal if open and show inline notice
            closeLowConfModal();
            showLowConfNotice('No low-confidence tracks remaining', 'info');
            return;
        }

        _currentCategory = _categories.length ? _categories[0] : null;
        _selectedSet.clear();

        if (openModal) openLowConfModal();
        renderCategorySlider();
        renderExpectedBatch();
        await renderUnclassifiedGrid();
    } catch (e) { hideLoading(); console.error('Error fetching low conf tracks', e); showLowConfNotice('Error fetching low-confidence tracks', 'error');
            // Defensive cleanup: ensure modal flag/state is consistent so UI can be re-opened later
            try { closeLowConfModal(); window.isReviewModalOpen = false; } catch (err) { /* ignore */ }
        }
}

function openLowConfModal() { const modal = document.getElementById('lowConfModal'); if (!modal) return; modal.style.display='block'; document.body.style.overflow='hidden'; window.isReviewModalOpen=true; }
function closeLowConfModal() { const modal = document.getElementById('lowConfModal'); if (!modal) return; modal.style.display='none'; document.body.style.overflow=''; window.isReviewModalOpen=false; }

// Ensure UI sync when modal is closed (refresh authoritative results and low-conf list)
async function handleLowConfClose() {
    closeLowConfModal();
    try {
        if (window.currentJobId) {
            await refreshResults(window.currentJobId);
            // Refresh low-conf data but do NOT re-open the modal after a user-initiated close
            await fetchLowConfTracks(window.currentJobId, false);
        }
    } catch (e) {
        console.warn('Error refreshing after modal close', e);
    }
}

function renderCategorySlider() {
    const container = document.getElementById('categorySlider'); container.innerHTML = '';
    if (!_categories.length) { const el=document.createElement('div'); el.style.color='#aaa'; el.textContent='No expected categories'; container.appendChild(el); return; }
    _categories.forEach(cat => { const btn = document.createElement('button'); btn.className='btn'; btn.style.minWidth='90px'; btn.style.flex='0 0 auto'; btn.textContent=`${cat} (${_lowConfByClass[cat].length})`; if (cat===_currentCategory) btn.classList.add('btn-primary'); btn.onclick = () => { _currentCategory = cat; _selectedSet.clear(); _batchIndexByCat[cat] = _batchIndexByCat[cat] || 0; renderCategorySlider(); renderExpectedBatch(); }; container.appendChild(btn); });
}

function renderExpectedBatch() {
    const grid = document.getElementById('expectedGrid'); const title = document.getElementById('expectedCategoryName'); const count = document.getElementById('expectedCount'); const indicator = document.getElementById('batchIndicator'); const removeBtn = document.getElementById('expectedRemoveBtn'); const confirmAllBtn = document.getElementById('expectedConfirmAllBtn');
    grid.innerHTML='';
    // responsive gallery: flow items by image width, small margin between items, no internal padding
    grid.style.display = 'flex';
    grid.style.flexWrap = 'wrap';
    grid.style.gap = '8px';
    grid.style.alignItems = 'flex-start';
    grid.style.padding = '0';
    if (!_currentCategory) { title.textContent='-'; count.textContent='0'; indicator.textContent='0 / 0'; removeBtn.style.display='none'; confirmAllBtn.style.display='none'; return; }
    const list = _lowConfByClass[_currentCategory]||[]; const totalBatches = Math.max(1, Math.ceil(list.length/_batchSize)); const idx = _batchIndexByCat[_currentCategory]||0; const start = idx*_batchSize; const batch = list.slice(start, start+_batchSize);
    title.textContent=_currentCategory; count.textContent=list.length; indicator.textContent=`${idx+1} / ${totalBatches}`;

    batch.forEach(rec => {
        const card = document.createElement('div');
        card.style.position = 'relative';
        card.style.background = '#02040a';
        card.style.borderRadius = '6px';
        card.style.overflow = 'hidden';
        card.style.padding = '0';
        card.style.cursor = 'pointer';
        card.style.display = 'inline-flex';
        card.style.flexDirection = 'column';
        card.style.justifyContent = 'flex-start';
        card.style.minHeight = 'auto';
        card.style.margin = '0';

        // Thumbnail container (fixed height, width follows image aspect)
        const thumb = document.createElement('div');
        thumb.style.width = 'auto';
        thumb.style.height = '140px';
        thumb.style.display = 'flex';
        thumb.style.alignItems = 'center';
        thumb.style.justifyContent = 'center';
        thumb.style.background = '#000';
        thumb.style.borderRadius = '4px';
        thumb.style.overflow = 'hidden';
        thumb.style.flex = '0 0 auto';

        const img = document.createElement('img');
        img.src = rec.image || '';
        img.style.height = '140px';
        img.style.width = 'auto';
        img.style.display = 'block';
        img.style.objectFit = 'contain';
        img.alt = `ID:${rec.track_id}`;
        thumb.appendChild(img);
        card.appendChild(thumb);

        // overlay for selection (no text overlay on image)
        const overlay = document.createElement('div'); overlay.style.position='absolute'; overlay.style.inset='6px'; overlay.style.borderRadius='6px'; overlay.style.background='rgba(0,0,0,0)'; overlay.style.transition='background 0.15s'; card.appendChild(overlay);
        const footer = document.createElement('div'); footer.style.display='flex'; footer.style.justifyContent='space-between'; footer.style.alignItems='center'; footer.style.marginTop='6px'; footer.style.width='100%'; footer.style.boxSizing='border-box';
        const info = document.createElement('div'); info.style.color='#ddd'; info.style.fontSize='12px'; info.innerHTML = `ID:${rec.track_id} • ${(rec.median_conf*100).toFixed(1)}%`; footer.appendChild(info);
        const chk = document.createElement('div'); chk.style.width='20px'; chk.style.height='20px'; chk.style.border='2px solid #444'; chk.style.borderRadius='4px'; chk.style.background='transparent'; footer.appendChild(chk);
        card.appendChild(footer);

        card.onclick = (e) => {
            if (_selectedSet.has(rec.track_id)) {
                _selectedSet.delete(rec.track_id); overlay.style.background='rgba(0,0,0,0)'; chk.style.background='transparent';
            } else {
                _selectedSet.add(rec.track_id); overlay.style.background='rgba(255,99,71,0.25)'; chk.style.background='rgba(255,99,71,0.9)';
            }
            // toggle Remove visibility
            removeBtn.style.display = (_selectedSet.size > 0) ? 'inline-flex' : 'none';
        };

        grid.appendChild(card);
    });
    removeBtn.style.display='none'; confirmAllBtn.style.display = batch.length ? 'inline-flex' : 'none';
}

// batch nav
document.getElementById('batchPrevBtn')?.addEventListener('click', () => { if (!_currentCategory) return; _batchIndexByCat[_currentCategory] = Math.max(0, (_batchIndexByCat[_currentCategory]||0)-1); renderExpectedBatch(); });
document.getElementById('batchNextBtn')?.addEventListener('click', () => { if (!_currentCategory) return; const list=_lowConfByClass[_currentCategory]||[]; const totalBatches=Math.max(1,Math.ceil(list.length/_batchSize)); _batchIndexByCat[_currentCategory]=Math.min(totalBatches-1, (_batchIndexByCat[_currentCategory]||0)+1); renderExpectedBatch(); });

// Remove selected -> unclassify immediately
document.getElementById('expectedRemoveBtn')?.addEventListener('click', async () => {
    const ids = Array.from(_selectedSet);
    if (!ids.length) return showLowConfNotice('Select images to remove to unclassified', 'info');
    if (!confirm(`Remove ${ids.length} selected image(s) to unclassified?`)) return;
    try {
        showLoading();
        for (const tid of ids) {
            const resp = await fetch(`/api/job/${window.currentJobId}/correct_track`, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ track_id: tid, action: 'unclassify' })});
            try { const data = await resp.json(); if (data && data.quality_metrics) updateQualityUIFromMetrics(data.quality_metrics, true); } catch(e){ console.warn('Could not parse unclassify response', e); }
        }
        hideLoading();
        _selectedSet.clear();
        await fetchLowConfTracks(window.currentJobId);
        await refreshResults(window.currentJobId);
        showLowConfNotice('Selected items moved to unclassified', 'success');
    } catch (e) { hideLoading(); console.error(e); showLowConfNotice('Failed to remove selected', 'error'); }
});

// Confirm all in current batch (set class + confidence=1)
document.getElementById('expectedConfirmAllBtn')?.addEventListener('click', async () => {
    if (!_currentCategory) return; const list = _lowConfByClass[_currentCategory] || []; const idx = _batchIndexByCat[_currentCategory] || 0; const start = idx * _batchSize; const batch = list.slice(start, start + _batchSize);
    if (!batch.length) return showLowConfNotice('Nothing to confirm in this batch', 'info');
    if (!confirm(`Confirm ${batch.length} images as "${_currentCategory}"?`)) return;
    const ids = batch.map(r => r.track_id);
    try {
        showLoading(); const resp = await fetch(`/api/job/${window.currentJobId}/correct_tracks`, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ track_ids: ids, new_class: _currentCategory })}); const data = await resp.json(); hideLoading(); if (data && data.success) { if (data.quality_metrics) updateQualityUIFromMetrics(data.quality_metrics, true); await fetchLowConfTracks(window.currentJobId); await refreshResults(window.currentJobId); showLowConfNotice(`Confirmed ${ids.length} tracks as ${_currentCategory}`, 'success'); } else { showLowConfNotice('Failed to confirm: ' + (data.error || 'unknown'), 'error'); } } catch (e) { hideLoading(); console.error(e); showLowConfNotice('Error confirming batch', 'error'); }
});

// Render unclassified grid
async function renderUnclassifiedGrid() {
    const grid = document.getElementById('unclassifiedGrid');
    const countEl = document.getElementById('unclassifiedCountModal');
    grid.innerHTML = '';

    // Build class list: combine expected categories, server vehicle_classes, and a sensible default list
    let classes = Array.from(new Set([...( _categories || [] )]));
    let vcData = null;
    try {
        const resp = await fetch(`/api/job/${window.currentJobId}/vehicle_classes`, {cache: 'no-store'});
        if (resp.ok) {
            vcData = await resp.json();
            const vcList = Object.entries(vcData.distribution || {}).sort((a,b)=>b[1]-a[1]).map(x=>x[0]);
            vcList.forEach(c => classes.push(c));
        }
    } catch (e) {
        console.warn('Could not fetch vehicle classes for unclassified selector', e);
    }

    // Add default common classes to ensure useful options (including requested ones)
    const DEFAULT_CLASSES = ['person','truck','bus','car','van','motorbike','bicycle','tuktuk','microbus','pickup','tricycle','unclassified'];
    DEFAULT_CLASSES.forEach(c => classes.push(c));

    // Deduplicate and order: prefer server-provided order (vcData) then expected categories, then defaults
    classes = Array.from(new Set(classes));

    // If we had a server distribution, keep its order first
    if (vcData && Object.keys(vcData.distribution || {}).length) {
        const serverOrder = Object.keys(vcData.distribution || {});
        classes.sort((a,b) => {
            const ai = serverOrder.indexOf(a); const bi = serverOrder.indexOf(b);
            if (ai !== -1 || bi !== -1) return (ai === -1) ? 1 : (bi === -1) ? -1 : (ai - bi);
            return a.localeCompare(b);
        });
    } else {
        classes.sort();
    }

    // Ensure 'unclassified' is present and at the end
    classes = classes.filter(c=>c !== 'unclassified');
    classes.push('unclassified');

    countEl.textContent = _unclassifiedList.length;

    // make unclassified grid a responsive gallery (cards size to image width)
    grid.style.display = 'flex';
    grid.style.flexWrap = 'wrap';
    grid.style.gap = '8px';
    grid.style.alignItems = 'flex-start';
    grid.style.padding = '0';

    _unclassifiedList.forEach(rec => {
        const card = document.createElement('div');
        card.style.position='relative';
        card.style.background='#02040a';
        card.style.borderRadius='6px';
        card.style.overflow='hidden';
        card.style.padding='0';
        card.style.display='inline-flex';
        card.style.flexDirection='column';
        card.style.justifyContent='flex-start';
        card.style.minHeight='auto';
        card.style.margin='0';

        const thumb = document.createElement('div');
        thumb.style.width = 'auto';
        thumb.style.height = '140px';
        thumb.style.display = 'flex';
        thumb.style.alignItems = 'center';
        thumb.style.justifyContent = 'center';
        thumb.style.background = '#000';
        thumb.style.borderRadius = '4px';
        thumb.style.overflow = 'hidden';
        thumb.style.flex = '0 0 auto';
        thumb.style.padding = '0';

        const img = document.createElement('img'); img.src = rec.image||''; img.style.height = '140px'; img.style.width = 'auto'; img.style.display = 'block'; img.style.objectFit = 'contain'; thumb.appendChild(img); card.appendChild(thumb);
        const footer = document.createElement('div'); footer.style.display='flex'; footer.style.justifyContent='space-between'; footer.style.alignItems='center'; footer.style.marginTop='6px';
        const info = document.createElement('div'); info.style.color='#ddd'; info.style.fontSize='12px'; info.innerHTML = `ID:${rec.track_id}`; footer.appendChild(info);
        const right = document.createElement('div'); right.style.display='flex'; right.style.gap='6px';

        // Per-image dropdown for selecting the corrected class (no single-image Confirm button)
        const sel = document.createElement('select');
        sel.className = 'unclass-select';
        sel.dataset.trackId = rec.track_id;
        sel.style.background = '#0b1220'; sel.style.color = '#fff'; sel.style.border = '1px solid #223'; sel.style.padding = '4px'; sel.style.borderRadius = '4px';
        classes.forEach(c => { const o = document.createElement('option'); o.value = c; o.textContent = c; sel.appendChild(o); });
        // default to current_class if available
        if (rec.current_class && classes.includes(rec.current_class)) sel.value = rec.current_class; else sel.value = 'unclassified';

        right.appendChild(sel); footer.appendChild(right); card.appendChild(footer); grid.appendChild(card);
    });
} 

// Unclassified Confirm All (reads per-image dropdowns and applies each selection)
document.getElementById('unclassifiedConfirmAllBtn')?.addEventListener('click', async () => {
    const selects = document.querySelectorAll('#unclassifiedGrid select.unclass-select');
    if (!selects || selects.length === 0) return showLowConfNotice('No unclassified items to confirm', 'info');
    const payloads = [];
    selects.forEach(s => { const tid = Number(s.dataset.trackId); const cls = s.value || 'unclassified'; payloads.push({ track_id: tid, new_class: cls }); });
    if (!confirm(`Confirm ${payloads.length} unclassified images with their selected classes?`)) return;
    const errors = [];
    try {
        showLoading();
        for (const p of payloads) {
            try {
                const resp = await fetch(`/api/job/${window.currentJobId}/correct_track`, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ track_id: p.track_id, new_class: p.new_class })});
                const data = await resp.json();
                if (!data || !data.success) {
                    errors.push({ track_id: p.track_id, error: data && data.error ? data.error : 'unknown' });
                }
            } catch (e) {
                errors.push({ track_id: p.track_id, error: String(e) });
            }
        }
        hideLoading();
        await fetchLowConfTracks(window.currentJobId);
        await refreshResults(window.currentJobId);
        if (errors.length === 0) {
            showLowConfNotice('All unclassified items confirmed', 'success');
        } else {
            console.error('Unclassified confirm errors:', errors);
            showLowConfNotice('Some items failed to confirm — check console for details', 'error');
        }
    } catch (e) { hideLoading(); console.error(e); showLowConfNotice('Error confirming all unclassified items', 'error'); }
});

// ensure modal reload pulls fresh data
document.getElementById('lowConfClose')?.addEventListener('click', () => handleLowConfClose());
document.getElementById('lowConfCloseBottom')?.addEventListener('click', () => handleLowConfClose());

// ==================== ROI Functions (multi-ROI) ====================

function setupROIHandlers() {
    const enableROI = document.getElementById('enableROI');
    const roiCanvasContainer = document.getElementById('roiCanvasContainer');
    const roiClearBtn = document.getElementById('roiClearBtn');
    const roiConfirmBtn = document.getElementById('roiConfirmBtn');
    const addNewROIBtn = document.getElementById('addNewROIBtn');
    const roiVideo = document.getElementById('roiVideo');
    const roiCanvas = document.getElementById('roiCanvas');

    enableROI.addEventListener('change', (e) => {
        roiEnabled = e.target.checked;
        roiCanvasContainer.style.display = roiEnabled ? 'block' : 'none';
        if (roiEnabled && currentVideoPath) {
            roiVideo.src = URL.createObjectURL(currentVideoFile);
            setTimeout(initROICanvas, 500);
        }
    });

    // Start drawing a new polygon - if a drawing is in progress, save it automatically when valid
    addNewROIBtn.addEventListener('click', (e) => {
        if (activeDrawingPoints && activeDrawingPoints.length >= 3) {
            // Auto-save current drawing before starting a new one
            saveCurrentROI();
            // Small delay to ensure UI updates, then start a fresh drawing
            setTimeout(() => startNewROI(), 100);
            return;
        }

        if (activeDrawingPoints && activeDrawingPoints.length > 0 && activeDrawingPoints.length < 3) {
            // Incomplete drawing: ask user whether to discard and start new
            const discard = confirm('Current drawing has fewer than 3 points and cannot be saved. Discard and start a new ROI?');
            if (discard) {
                clearCurrentDrawing();
                startNewROI();
            } else {
                // User chose to continue drawing
            }
            return;
        }

        // Normal start when no drawing in progress
        startNewROI();
    });

    roiClearBtn.addEventListener('click', clearCurrentDrawing);
    roiConfirmBtn.addEventListener('click', saveCurrentROI);

    // حفظ reference للـ canvas
    window.roiCanvasElement = roiCanvas;
}

function startNewROI() {
    if (!roiEnabled) return alert('Enable region filtering first');
    drawingMode = true;
    activeDrawingPoints = [];
    document.getElementById('roiCanvasContainer').style.display = 'block';
    document.getElementById('roiPointsDisplay').style.display = 'none';
    initROICanvas();
}

function initROICanvas() {
    const roiCanvas = document.getElementById('roiCanvas');
    const roiVideo = document.getElementById('roiVideo');

    roiCanvas.width = roiVideo.offsetWidth;
    roiCanvas.height = roiVideo.offsetHeight;
    window.roiCtx = roiCanvas.getContext('2d');
    roiCanvas.style.display = 'block';

    // رسم الـ video على canvas
    window.roiCtx.drawImage(roiVideo, 0, 0, roiCanvas.width, roiCanvas.height);

    // Ensure previous click handlers removed before adding new
    roiCanvas.replaceWith(roiCanvas.cloneNode(true));
    const freshCanvas = document.getElementById('roiCanvas');
    window.roiCanvasElement = freshCanvas;
    window.roiCtx = freshCanvas.getContext('2d');

    freshCanvas.addEventListener('click', (e) => {
        const rect = freshCanvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) * (freshCanvas.width / rect.width);
        const y = (e.clientY - rect.top) * (freshCanvas.height / rect.height);

        // close polygon if click near first point
        if (activeDrawingPoints.length > 2 && Math.abs(x - activeDrawingPoints[0].x) < 15 && Math.abs(y - activeDrawingPoints[0].y) < 15) {
            finishROIDrawing();
            return;
        }

        activeDrawingPoints.push({x, y});
        drawROICanvas();
    });

    // Redraw any existing saved polygons as overlay (subtle styling)
    drawROISavedPolygons();
}

function drawROICanvas() {
    const roiCanvas = window.roiCanvasElement;
    const roiVideo = document.getElementById('roiVideo');
    const ctx = window.roiCtx;

    // رسم الـ video
    ctx.drawImage(roiVideo, 0, 0, roiCanvas.width, roiCanvas.height);

    // Draw existing saved polygons for context
    drawROISavedPolygons();

    if (activeDrawingPoints.length === 0) return;

    // رسم خطوط المضلع current drawing
    ctx.strokeStyle = '#43e97b';
    ctx.lineWidth = 2;
    ctx.fillStyle = 'rgba(67, 233, 123, 0.12)';

    ctx.beginPath();
    ctx.moveTo(activeDrawingPoints[0].x, activeDrawingPoints[0].y);

    for (let i = 1; i < activeDrawingPoints.length; i++) {
        ctx.lineTo(activeDrawingPoints[i].x, activeDrawingPoints[i].y);
    }

    if (activeDrawingPoints.length > 1) ctx.lineTo(activeDrawingPoints[0].x, activeDrawingPoints[0].y);

    ctx.stroke();
    ctx.fill();

    // draw points
    activeDrawingPoints.forEach((point, i) => {
        ctx.fillStyle = i === 0 ? '#ff6b6b' : '#43e97b';
        ctx.beginPath();
        ctx.arc(point.x, point.y, 6, 0, Math.PI * 2);
        ctx.fill();
    });
}

function drawROISavedPolygons() {
    const roiCanvas = window.roiCanvasElement;
    if (!roiCanvas || !window.roiCtx) return;
    const ctx = window.roiCtx;
    const roiVideo = document.getElementById('roiVideo');
    ctx.drawImage(roiVideo, 0, 0, roiCanvas.width, roiCanvas.height);

    // draw saved polygons lightly
    roiPolygons.forEach((poly, idx) => {
        const points = poly.points;
        if (!points || points.length === 0) return;
        ctx.beginPath();
        ctx.moveTo(points[0].x, points[0].y);
        for (let i = 1; i < points.length; i++) ctx.lineTo(points[i].x, points[i].y);
        ctx.closePath();
        ctx.strokeStyle = 'rgba(102,126,234,0.6)';
        ctx.lineWidth = 1.5;
        ctx.fillStyle = 'rgba(102,126,234,0.06)';
        ctx.stroke();
        ctx.fill();
    });
}

function clearCurrentDrawing() {
    activeDrawingPoints = [];
    drawingMode = false;
    document.getElementById('roiPointsDisplay').style.display = 'none';
    if (window.roiCtx && window.roiCanvasElement) {
        window.roiCtx.drawImage(document.getElementById('roiVideo'), 0, 0, window.roiCanvasElement.width, window.roiCanvasElement.height);
    }
}

function finishROIDrawing() {
    if (activeDrawingPoints.length < 3) {
        alert('يجب تحديد 3 نقاط على الأقل لرسم المنطقة');
        return;
    }
    drawingMode = false;
    // user may save the polygon using 'Save ROI'
}

function saveCurrentROI() {
    if (activeDrawingPoints.length < 3) return alert('No ROI to save');
    const newRoi = { id: nextRoiId++, points: activeDrawingPoints.slice() };
    roiPolygons.push(newRoi);
    activeDrawingPoints = [];
    drawingMode = false;
    document.getElementById('roiCanvasContainer').style.display = 'none';
    renderROIList();
}

function renderROIList() {
    const roiList = document.getElementById('roiList');
    const container = document.getElementById('roiListContainer');
    if (!roiList || !container) return;
    roiList.innerHTML = '';
    if (!roiPolygons || roiPolygons.length === 0) { container.style.display = 'none'; return; }
    container.style.display = 'block';

    roiPolygons.forEach((poly) => {
        const card = document.createElement('div');
        card.style.cssText = 'background:#0b1220; padding:8px 10px; border-radius:8px; display:flex; align-items:center; gap:8px; color:#ddd;';
        card.innerHTML = `<div style="font-weight:600;">ROI #${poly.id}</div><div style="font-size:12px; color:#aaa;">(${poly.points.length} pts)</div>`;
        const del = document.createElement('button');
        del.className = 'btn btn-danger';
        del.style.fontSize = '12px';
        del.textContent = 'Delete';
        del.addEventListener('click', () => { deleteROI(poly.id); });
        card.appendChild(del);
        roiList.appendChild(card);
    });
}

function deleteROI(id) {
    roiPolygons = roiPolygons.filter(p => p.id !== id);
    renderROIList();
}
