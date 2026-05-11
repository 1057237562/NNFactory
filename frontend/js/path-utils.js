/**
 * Shared path utilities - extracted from connections.js and canvas.js
 */
function createBezierPath(x1, y1, x2, y2) {
    return `M ${x1} ${y1} C ${x1 + 50} ${y1}, ${x2 - 50} ${y2}, ${x2} ${y2}`;
}

window.PathUtils = {
    createBezierPath
};