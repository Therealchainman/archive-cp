// ==UserScript==
// @name         New Userscript
// @namespace    http://tampermonkey.net/
// @version      0.1
// @description  try to take over the world!
// @author       You
// @match        https://leetcode.com/*
// @icon         https://www.google.com/s2/favicons?domain=leetcode.com
// @grant        none
// ==/UserScript==

(function() {
    const s = document.createElement('style')
    s.textContent=".css-1ubm0bb-Value { max-height: none }"
    document.body.appendChild(s)
})();
