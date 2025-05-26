// --- Constants and Configuration ---
const TARGET_SPAN_CLASS = "yt-core-attributed-string--white-space-pre-wrap";
const PROCESSED_ATTRIBUTE = 'data-observer-processed';
const API_ENDPOINT = 'http://localhost:8000/process_text';
const PLACEHOLDER_TEXT = ""; // Text to show while waiting for API (e.g., "..." or "")

// --- API Interaction ---
/**
 * Sends text to the backend API for processing and updates the spanNode.
 * @param {string} originalText The text to process.
 * @param {HTMLElement} spanNodeToUpdate The DOM element whose text content should be updated.
 */
async function processTextWithAPI(originalText, spanNodeToUpdate) {
    console.log(`Sending to API: "${originalText.substring(0, 50)}..."`); // Log snippet
    try {
        const response = await fetch(API_ENDPOINT, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                // Add any other headers your API might require, like an API key
            },
            body: JSON.stringify({ text: originalText }),
        });

        if (!response.ok) {
            // Handle HTTP errors (e.g., 400, 404, 500)
            const errorText = await response.text();
            console.error(`API Error: ${response.status} ${response.statusText}. Response: ${errorText}`);
            // Display an error message or revert to original text
            spanNodeToUpdate.textContent = `[Error: API ${response.status}]`;
            return;
        }

        const data = await response.json();

        if (data && typeof data.processed_text === 'string') {
            spanNodeToUpdate.textContent = data.processed_text;
            console.log(`API Response for "${originalText.substring(0, 30)}...": "${data.processed_text}"`);
        }
        else {
            console.error("API response format unexpected:", data);
            spanNodeToUpdate.textContent = "[Error: Invalid API Response Format]";
        }

    } catch (error) {
        // Handle network errors 
        console.error("Error calling API:", error);
        // Display an error message or revert to original text
        spanNodeToUpdate.textContent = "[Error: Network/Fetch Failed]";
    }
}


// --- DOM Interaction & Mutation Observer ---
/**
 * Handles the appearance of a new target span element.
 * @param {HTMLElement} spanNode The detected span element.
 */
async function handleNewTargetSpanAppeared(spanNode) {
    // Check if this node has already been processed or is actively being processed
    if (spanNode.hasAttribute(PROCESSED_ATTRIBUTE)) {
        return;
    }
    spanNode.setAttribute(PROCESSED_ATTRIBUTE, 'true'); // Mark as processing initiated

    const originalText = spanNode.textContent || "";

    // Avoid processing empty or whitespace-only spans, or our own placeholders
    if (!originalText.trim() || originalText === PLACEHOLDER_TEXT) {
        spanNode.removeAttribute(PROCESSED_ATTRIBUTE); // Unmark if not actually processed
        return;
    }

    let elementId = spanNode.getAttribute('data-sim-id') ||
                    spanNode.id ||
                    `dyn-span-${Date.now()}-${Math.random().toString(16).slice(2)}`;

    const commentViewModel = spanNode.closest('ytd-comment-view-model');
    if (commentViewModel) {
        const timeLink = commentViewModel.querySelector('#published-time-text a');
        if (timeLink && timeLink.href) {
            try {
                const url = new URL(timeLink.href, window.location.origin);
                const lcId = url.searchParams.get('lc');
                if (lcId) {
                    elementId = lcId; // Use YouTube's comment ID if available
                }
            } catch (e) {
                console.warn('Could not parse comment link URL for a stable ID:', e);
            }
        }
    }

    console.log(`Target span detected (ID: ${elementId}). Original text: "${originalText.substring(0, 100)}..."`);

    // Set placeholder text immediately while waiting for the API
    const initialSpanHeight = spanNode.offsetHeight; 
    spanNode.textContent = PLACEHOLDER_TEXT;
    if (PLACEHOLDER_TEXT === "" && initialSpanHeight > 0) { // Attempt to prevent collapse if placeholder is empty
        spanNode.style.minHeight = `${initialSpanHeight}px`;
    }


    // Process the text with the API
    await processTextWithAPI(originalText, spanNode);

    // Clean up minHeight style if it was set
    if (PLACEHOLDER_TEXT === "" && initialSpanHeight > 0) {
        spanNode.style.minHeight = '';
    }
}

// Configuration for the observer:
// We want to observe child additions anywhere in the document's body and its subtree.
const observerConfig = {
    childList: true, // Observe additions or removals of child nodes
    subtree: true    // Observe changes in the entire subtree of document.body
};

// Callback function to execute when mutations are observed
const mutationCallback = function(mutationsList, observer) {
    for (const mutation of mutationsList) {
        if (mutation.type === 'childList') {
            mutation.addedNodes.forEach(node => {
                // Check if the added node is an ELEMENT_NODE
                if (node.nodeType === Node.ELEMENT_NODE) {
                    // Case 1: The added node itself is the target span
                    if (node.tagName.toLowerCase() === 'span' &&
                        node.classList.contains(TARGET_SPAN_CLASS)) {
                        handleNewTargetSpanAppeared(node);
                    }
                    // Case 2: The added node is a container, look for target spans inside it
                    // This is important if spans are added as part of a larger chunk of HTML
                    const nestedSpans = node.querySelectorAll(`span.${TARGET_SPAN_CLASS.split(' ').join('.')}`);
                    nestedSpans.forEach(span => handleNewTargetSpanAppeared(span));
                }
            });
        }
    }
};

// Create an observer instance linked to the callback function
const observer = new MutationObserver(mutationCallback);

// Start observing document.body for configured mutations
observer.observe(document.body, observerConfig);
console.log(`MutationObserver is now watching document.body for span elements with class '${TARGET_SPAN_CLASS}'.`);
