/**
 * Modal Loader — loads HTML partials asynchronously before app initialization.
 * All modals inject their content into the matching container element.
 */
(async function loadModals() {
    const modalMap = {
        'codeModal': 'html/modals/code-modal.html',
        'trainModal': 'html/modals/train-modal.html',
        'evalModal': 'html/modals/eval-modal.html',
        'datasetModal': 'html/modals/dataset-modal.html',
        'ppNodeModal': 'html/modals/pp-node-modal.html',
        'ppResultModal': 'html/modals/pp-result-modal.html',
        'weightsModal': 'html/modals/weights-modal.html',
    };
    try {
        await Promise.all(Object.entries(modalMap).map(async ([id, url]) => {
            const resp = await fetch(url);
            if (!resp.ok) throw new Error(`Failed to load ${url}: ${resp.status}`);
            const target = document.querySelector(`#${id} > .modal-content`);
            if (target) target.innerHTML = await resp.text();
        }));
        // Load sidebar layers partial
        const sidebarResp = await fetch('html/sidebar-layers.html');
        if (sidebarResp.ok) {
            const sidebarTarget = document.getElementById('layerItems');
            if (sidebarTarget) sidebarTarget.innerHTML = await sidebarResp.text();
        }
        window.modalsLoaded = true;
        window.dispatchEvent(new Event('modals-loaded'));
    } catch (err) {
        console.error('Modal loader error:', err);
        // Still dispatch event so app can proceed (graceful degradation)
        window.modalsLoaded = false;
        window.dispatchEvent(new Event('modals-loaded'));
    }
})();
