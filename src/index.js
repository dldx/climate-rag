// Import all required packages
import 'htmx.org';
import DOMPurify from 'dompurify';
import { render } from 'katex';
import { marked } from 'marked';
import markedKatex from 'marked-katex-extension';

// Initialize marked with KaTeX extension
marked.use(markedKatex());

// Make modules available globally
window.DOMPurify = DOMPurify;
window.marked = marked;
window.katex = { render };

marked.use(markedKatex({
    throwOnError: true,

}));

function renderMarkdownWithKatex(text) {
    var el = document.createElement('template');
    el.innerHTML = DOMPurify.sanitize(marked.parse(escapeBrackets(text)))
    el.content.querySelectorAll('a').forEach(link => {
        link.setAttribute('target', '_blank')
    })
    document.currentScript.parentElement.innerHTML = el.innerHTML;
}
window.renderMarkdownWithKatex = renderMarkdownWithKatex;

function escapeBrackets(text) {
    const pattern =
        /(```[\s\S]*?```|`.*?`)|\\\[([\s\S]*?[^\\])\\\]|\\\((.*?)\\\)/g;
    return text.replace(
        pattern,
        (match, codeBlock, squareBracket, roundBracket) => {
            if (codeBlock) {
                return codeBlock;
            } else if (squareBracket) {
                return `$$${squareBracket}$$`;
            } else if (roundBracket) {
                return `$${roundBracket}$`;
            }
            return match;
        },
    );
}

window.escapeBrackets = escapeBrackets;