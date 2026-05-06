/**
 * Shared path utilities - extracted from connections.js and canvas.js
 */
function createBezierPath(x1, y1, x2, y2) {
    const dx = Math.abs(x2 - x1);
    const cp = Math.max(50, dx * 0.5);
    return `M ${x1} ${y1} C ${x1 + cp} ${y1}, ${x2 - cp} ${y2}, ${x2} ${y2}`;
}

window.PathUtils = {
    createBezierPath
};