(function () {
    const generateBtn = document.getElementById('generateBtn');
    const rowsInput = document.getElementById('rowsInput');
    const statusEl = document.getElementById('status');
    const tableWrap = document.getElementById('tableWrap');
    const tableHead = document.getElementById('tableHead');
    const tableBody = document.getElementById('tableBody');
    const resultsInfo = document.getElementById('resultsInfo');

    function showStatus(text, busy) {
        if (statusEl) statusEl.textContent = text;
        if (generateBtn) generateBtn.disabled = !!busy;
        if (generateBtn) generateBtn.textContent = busy ? 'Generating...' : 'Generate';
    }

    function validateRows(n) {
        n = Number(n);
        if (isNaN(n)) return false;
        if (n < 5) return false;
        if (n > 30) return false;
        return true;
    }

    function renderTable(columns, rows) {
        if (!tableHead || !tableBody || !resultsInfo || !tableWrap) return;

        tableHead.innerHTML = '';
        tableBody.innerHTML = '';

        const trHead = document.createElement('tr');
        columns.forEach(function (col) {
            const th = document.createElement('th');
            th.textContent = col;
            trHead.appendChild(th);
        });
        tableHead.appendChild(trHead);

        rows.forEach(function (r) {
            const tr = document.createElement('tr');
            r.forEach(function (cell) {
                const td = document.createElement('td');
                td.textContent = String(cell);
                tr.appendChild(td);
            });
            tableBody.appendChild(tr);
        });

        resultsInfo.textContent = 'Showing ' + rows.length + ' rows — model output preview';
        tableWrap.hidden = false;
    }

    function callGenerateAPI(model, rows, options) {
        options = options || {};
        var timeout = options.timeout || 60000;

        return new Promise(function (resolve, reject) {
            var xhr = new XMLHttpRequest();
            xhr.open('POST', '/api/generate', true);
            xhr.setRequestHeader('Content-Type', 'application/json;charset=UTF-8');
            xhr.timeout = timeout;

            xhr.onreadystatechange = function () {
                if (xhr.readyState !== 4) return;

                if (xhr.status >= 200 && xhr.status < 300) {
                    try {
                        var data = JSON.parse(xhr.responseText);
                        resolve(data);
                    } catch (e) {
                        reject(new Error('Invalid JSON from server'));
                    }
                } else {
                    var msg = 'HTTP ' + xhr.status;
                    try { msg += ' - ' + xhr.responseText; } catch (_) {}
                    reject(new Error(msg));
                }
            };

            xhr.ontimeout = function () {
                reject(new Error('Request timed out'));
            };

            xhr.onerror = function () {
                reject(new Error('Network error'));
            };

            var payload = { model: model, rows: rows };
            xhr.send(JSON.stringify(payload));
        });
    }

    function getSelectedModel() {
        var sel = document.querySelector('input[name="model"]:checked');
        return sel ? sel.value : 'gan';
    }

    function handleGenerateClick() {
        var rows = Number(rowsInput.value);

        if (!validateRows(rows)) {
            alert('Rows must be a number between 5 and 30.');
            return;
        }

        var model = getSelectedModel();
        showStatus('Running', true);

        callGenerateAPI(model, rows)
            .then(function (resp) {
                if (!resp || !Array.isArray(resp.columns) || !Array.isArray(resp.rows)) {
                    throw new Error('Server returned unexpected format');
                }
                renderTable(resp.columns, resp.rows);
                showStatus('Idle', false);
            })
            .catch(function (err) {
                console.error('Generate API error:', err);
                showStatus('Error', false);
                tableWrap.hidden = true;
                if (resultsInfo) {
                    resultsInfo.textContent = 'Error: could not generate data. See console for details.';
                }
            });
    }

    function init() {
        if (!generateBtn || !rowsInput) return;

        generateBtn.addEventListener('click', handleGenerateClick);

        showStatus('Idle', false);

        if (tableWrap) tableWrap.hidden = true;
        if (resultsInfo) resultsInfo.textContent = 'No data yet.';
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
