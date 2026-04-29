// State management
let selectedSourceFile = null;
let selectedTemplateFile = null;
let isProcessing = false;

// DOM Elements
const sourceInput = document.getElementById('source-input');
const sourceDropZone = document.getElementById('source-drop-zone');
const sourcePreview = document.getElementById('source-preview');

const customTemplateInput = document.getElementById('custom-template-input');
const uploadCustomBtn = document.getElementById('upload-custom-btn');
const templateGrid = document.getElementById('template-grid');
const customTemplatePreview = document.getElementById('custom-template-preview');
const customTemplatePreviewContainer = document.getElementById('selected-template-preview-container');

const swapBtn = document.getElementById('swap-btn');
const enhanceCheckbox = document.getElementById('enhance-checkbox');
const resultImg = document.getElementById('result-img');
const resultPlaceholder = document.getElementById('result-placeholder');
const resultActions = document.getElementById('result-actions');
const downloadBtn = document.getElementById('download-btn');
const resetBtn = document.getElementById('reset-btn');
const errorMessage = document.getElementById('error-message');
const warningsContainer = document.getElementById('warnings-container');

// Initialization
document.addEventListener('DOMContentLoaded', () => {
    fetchTemplates();
});

// Event Listeners
sourceDropZone.addEventListener('click', () => sourceInput.click());
sourceInput.addEventListener('change', (e) => handleFileSelect(e.target.files[0], 'source'));

uploadCustomBtn.addEventListener('click', () => customTemplateInput.click());
customTemplateInput.addEventListener('change', (e) => handleFileSelect(e.target.files[0], 'template-custom'));

swapBtn.addEventListener('click', handleSwap);
resetBtn.addEventListener('click', resetApp);

// Drag and Drop
sourceDropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    sourceDropZone.classList.add('active');
});

sourceDropZone.addEventListener('dragleave', () => {
    sourceDropZone.classList.remove('active');
});

sourceDropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    sourceDropZone.classList.remove('active');
    handleFileSelect(e.dataTransfer.files[0], 'source');
});

// Functions
async function fetchTemplates() {
    try {
        const response = await fetch('/templates');
        const data = await response.json();
        renderTemplates(data.templates);
    } catch (err) {
        console.error('Failed to fetch templates:', err);
        templateGrid.innerHTML = '<div class="error-text">Failed to load templates.</div>';
    }
}

function renderTemplates(templates) {
    if (templates.length === 0) {
        templateGrid.innerHTML = '<div class="loader">No templates found in /templates folder.</div>';
        return;
    }

    templateGrid.innerHTML = '';
    templates.forEach(filename => {
        const item = document.createElement('div');
        item.className = 'template-item';
        item.innerHTML = `<img src="/templates/${filename}" alt="${filename}">`;
        item.addEventListener('click', () => selectTemplate(filename, item));
        templateGrid.appendChild(item);
    });
}

async function selectTemplate(filename, element) {
    // Clear custom preview if it exists
    customTemplatePreviewContainer.classList.add('hidden');

    // UI feedback
    document.querySelectorAll('.template-item').forEach(el => el.classList.remove('selected'));
    element.classList.add('selected');

    // Fetch the template as a blob to use in FormData
    try {
        const response = await fetch(`/templates/${filename}`);
        const blob = await response.blob();
        selectedTemplateFile = new File([blob], filename, { type: blob.type });
        updateSwapButton();
    } catch (err) {
        showError('Failed to select template.');
    }
}

function handleFileSelect(file, type) {
    if (!file) return;
    if (!file.type.startsWith('image/')) {
        showError('Please select an image file.');
        return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        if (type === 'source') {
            selectedSourceFile = file;
            sourcePreview.src = e.target.result;
            sourcePreview.classList.remove('hidden');
        } else if (type === 'template-custom') {
            selectedTemplateFile = file;
            customTemplatePreview.src = e.target.result;
            customTemplatePreviewContainer.classList.remove('hidden');
            // Deselect grid templates
            document.querySelectorAll('.template-item').forEach(el => el.classList.remove('selected'));
        }
        updateSwapButton();
    };
    reader.readAsDataURL(file);
}

function updateSwapButton() {
    swapBtn.disabled = !selectedSourceFile || !selectedTemplateFile || isProcessing;
}

async function handleSwap() {
    if (isProcessing) return;

    setProcessing(true);
    clearError();
    clearWarnings();
    resultPlaceholder.classList.remove('hidden');
    resultImg.classList.add('hidden');
    resultActions.classList.add('hidden');

    const formData = new FormData();
    formData.append('source_image', selectedSourceFile);
    formData.append('target_image', selectedTemplateFile);
    formData.append('enhance', enhanceCheckbox.checked);

    try {
        const response = await fetch('/swap', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            resultImg.src = 'data:image/png;base64,' + data.result;
            resultImg.classList.remove('hidden');
            resultPlaceholder.classList.add('hidden');
            resultActions.classList.remove('hidden');
            downloadBtn.href = resultImg.src;

            if (data.warnings && data.warnings.length > 0) {
                showWarnings(data.warnings);
            }
        } else {
            showError(data.error || 'Failed to perform swap.');
        }
    } catch (err) {
        showError('Connection error. Is the server running?');
        console.error(err);
    } finally {
        setProcessing(false);
    }
}

function setProcessing(val) {
    isProcessing = val;
    swapBtn.disabled = val;
    swapBtn.textContent = val ? 'Processing... (~10s)' : 'SWAP FACES';
    updateSwapButton();
}

function showError(msg) {
    errorMessage.textContent = msg;
    errorMessage.classList.remove('hidden');
}

function clearError() {
    errorMessage.textContent = '';
    errorMessage.classList.add('hidden');
}

function showWarnings(warnings) {
    warningsContainer.innerHTML = '';
    warnings.forEach(w => {
        const banner = document.createElement('div');
        banner.className = 'warning-banner';
        banner.textContent = '⚠️ ' + w;
        warningsContainer.appendChild(banner);
    });
}

function clearWarnings() {
    warningsContainer.innerHTML = '';
}

function resetApp() {
    resultImg.src = '';
    resultImg.classList.add('hidden');
    resultPlaceholder.classList.remove('hidden');
    resultActions.classList.add('hidden');
    clearWarnings();
    clearError();
}
