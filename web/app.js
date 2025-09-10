const DGM_META = {
    "linear_gaussian": {
        name: "Linear Gaussian",
        equation: `\\begin{align}
        X &= \\alpha^T Z + \\varepsilon_1 \\\\
        Y &= \\beta^T Z + \\text{effect\\_size} \\cdot X + \\varepsilon_2
        \\end{align}
        \\text{where } \\varepsilon_1, \\varepsilon_2 \\sim \\mathcal{N}(0, \\sigma^2) \\text{ (Gaussian noise)}`
    },
    "nonlinear_gaussian": {
        name: "Nonlinear Gaussian",
        equation: `\\begin{align}
        X &= \\sin(\\text{effect\\_size} \\cdot \\sum_j Z_j) + \\varepsilon_1 \\\\
        Y &= \\exp(\\text{effect\\_size} \\cdot \\sum_j Z_j \\cdot 0.2) + \\varepsilon_2
        \\end{align}
        \\text{where } \\varepsilon_1, \\varepsilon_2 \\sim \\mathcal{N}(0, \\sigma^2) \\text{ (Gaussian noise)}`
    },
    "discrete_categorical": {
        name: "Discrete Categorical",
        equation: `\\begin{align}
        Z_j &\\sim \\text{DiscreteUniform}(0, n_{\\text{categories}}-1) \\\\
        X &= \\sum_j Z_j + \\varepsilon_1 \\\\
        Y &= \\sum_j Z_j + \\varepsilon_2
        \\end{align}
        \\text{where } \\varepsilon_1, \\varepsilon_2 \\sim \\mathcal{N}(0, \\sigma^2) \\text{ (Gaussian noise)}`
    },
    "mixed_data": {
        name: "Mixed Data",
        equation: `\\begin{align}
        X &= Z^T \\alpha + \\varepsilon_1 \\\\
        Y &= Z^T \\beta + \\text{effect\\_size} \\cdot X + \\varepsilon_2
        \\end{align}
        \\text{where } \\varepsilon_1, \\varepsilon_2 \\sim \\mathcal{N}(0, \\sigma^2) \\text{ (Gaussian noise)}`
    },
    "non_gaussian_continuous": {
        name: "Non-Gaussian Continuous",
        equation: `\\begin{align}
        X &= |Z^T \\alpha| + \\varepsilon_1 \\\\
        Y &= (Z^T \\beta)^2 + \\text{effect\\_size} \\cdot X + \\varepsilon_2
        \\end{align}
        \\text{where } \\varepsilon_1, \\varepsilon_2 \\sim \\text{Exponential}(\\lambda = 1.0) \\text{ (Exponential noise)}`
    }
};

let allData = [];
let dgmMap = {};
let chartCalibration, chartPower;

const dgmSelect = document.getElementById('dgm-select');
const sampleSizeSelect = document.getElementById('sample-size-select');
const csvInput = document.getElementById('csv-input');
const dgmEquationEl = document.getElementById('dgm-equation');

const DEFAULT_CSV = "results/ci_benchmark_summaries.csv";

window.addEventListener('DOMContentLoaded', () => {
    fetch(DEFAULT_CSV)
        .then(res => {
            if (!res.ok) throw new Error("Default CSV not found");
            return res.text();
        })
        .then(csvText => {
            allData = parseCSV(csvText);
            buildDGMMap(allData);
            populateDropdowns();
            renderCharts();
        })
        .catch(() => {
            // file not found, don't crash
        });
});

csvInput.addEventListener('change', handleCSVUpload);
dgmSelect.addEventListener('change', onDGMChange);
sampleSizeSelect.addEventListener('change', renderCharts);

function handleCSVUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = e => {
        allData = parseCSV(e.target.result);
        buildDGMMap(allData);
        populateDropdowns();
        renderCharts();
    };
    reader.readAsText(file);
}

function parseCSV(csvText) {
    const [header, ...rows] = csvText.trim().split('\n');
    const keys = header.split(',');
    return rows.map(row =>
        Object.fromEntries(
            row.split(',').map((val, i) => [keys[i], parseValue(val)])
        )
    );
}

function parseValue(v) {
    if (!isNaN(parseFloat(v)) && isFinite(v)) return parseFloat(v);
    if (v === 'True') return true;
    if (v === 'False') return false;
    return v;
}

function buildDGMMap(data) {
    dgmMap = {};
    data.forEach(row => {
        const { dgm, sample_size } = row;
        if (!dgmMap[dgm]) dgmMap[dgm] = new Set();
        dgmMap[dgm].add(row.sample_size);
    });
    Object.keys(dgmMap).forEach(dgm => {
        dgmMap[dgm] = Array.from(dgmMap[dgm]).sort((a, b) => a - b);
    });
}

function setSelectOptions(select, list) {
    select.innerHTML = '';
    for (const obj of list) {
        const option = document.createElement('option');
        option.value = obj.value;
        option.textContent = obj.label;
        select.appendChild(option);
    }
}

function populateDropdowns() {
    setSelectOptions(dgmSelect, Object.keys(dgmMap).map(d => ({ value: d, label: DGM_META[d]?.name || d })));
    onDGMChange();
}

function onDGMChange() {
    const dgm = dgmSelect.value;
    const sizes = dgm ? dgmMap[dgm] : [];
    setSelectOptions(sampleSizeSelect, sizes.map(s => ({ value: s, label: s })));
    updateDGMEquation();
    renderCharts();
}

function updateDGMEquation() {
    const dgm = dgmSelect.value;
    dgmEquationEl.innerHTML = DGM_META[dgm]?.equation || '';
    if (window.MathJax) MathJax.typesetPromise([dgmEquationEl]);
}

function computeStats(data, groupBy, valueCol) {
    const grouped = {};
    data.forEach(row => {
        const key = groupBy(row);
        if (!grouped[key]) grouped[key] = [];
        grouped[key].push(row[valueCol]);
    });
    
    return Object.entries(grouped).map(([key, values]) => {
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
        const stderr = Math.sqrt(variance / values.length);
        return {
            key: parseFloat(key),
            mean,
            stderr,
            upper: mean + stderr,
            lower: Math.max(0, mean - stderr)
        };
    }).sort((a, b) => a.key - b.key);
}

function renderCharts() {
    if (!allData.length) return;
    const dgm = dgmSelect.value;
    const sampleSize = Number(sampleSizeSelect.value);

    // Calibration plot: Type I error vs significance_level, effect_size == 0
    const calibData = allData.filter(r => r.dgm === dgm && r.sample_size === sampleSize && r.effect_size === 0);
    const ciTestsCalib = Array.from(new Set(calibData.map(r => r.ci_test)));
    
    const calibDatasets = ciTestsCalib.map((ciTest, idx) => {
        const testData = calibData.filter(r => r.ci_test === ciTest);
        const stats = computeStats(testData, row => row.significance_level, 'type1_error');
        
        return [
            {
                label: `${ciTest}`,
                data: stats.map(s => ({ x: s.key, y: s.mean })),
                borderColor: chartColor(idx, 1),
                backgroundColor: chartColor(idx, 0.2),
                pointRadius: 4,
                fill: false,
                tension: 0.2
            },
            // Added ribbon plot for standard error
            {
                label: `${ciTest} - Error Band`,
                data: stats.map(s => ({ x: s.key, y: s.upper })),
                borderColor: chartColor(idx, 0),
                backgroundColor: chartColor(idx, 0.15),
                fill: '+1',
                pointRadius: 0,
                tension: 0.2
            },
            {
                label: `${ciTest} - Lower`,
                data: stats.map(s => ({ x: s.key, y: s.lower })),
                borderColor: chartColor(idx, 0),
                backgroundColor: chartColor(idx, 0.15),
                fill: false,
                pointRadius: 0,
                tension: 0.2
            }
        ];
    }).flat();

    if (chartCalibration) chartCalibration.destroy();
    chartCalibration = new Chart(document.getElementById('calibration-plot').getContext('2d'), {
        type: 'line',
        data: {
            datasets: calibDatasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            aspectRatio: 1.6, 
            plugins: { 
                legend: { 
                    display: true,
                    filter: (legendItem) => !legendItem.text.includes('Band') && !legendItem.text.includes('Lower')
                } 
            },
            scales: {
                x: { 
                    title: { display: true, text: 'Significance Level' }, 
                    min: 0, 
                    max: 1,
                    type: 'linear'
                },
                y: { 
                    title: { display: true, text: 'Type I Error' }, 
                    min: 0, 
                    max: 1 
                }
            }
        }
    });

    // Power plot: Power vs effect size, significance_level == 0.05
    const powerData = allData.filter(row =>
        row.dgm === dgm &&
        row.sample_size === sampleSize &&
        row.significance_level === 0.05
    );
    
    const ciTestsPower = Array.from(new Set(powerData.map(r => r.ci_test)));
    const powerDatasets = ciTestsPower.map((ciTest, idx) => {
        const testData = powerData.filter(r => r.ci_test === ciTest);
        const stats = computeStats(testData, row => row.effect_size, 'power');
        
        return [
            {
                label: `${ciTest}`,
                data: stats.map(s => ({ x: s.key, y: s.mean })),
                borderColor: chartColor(idx, 1),
                backgroundColor: chartColor(idx, 0.2),
                pointRadius: 3,
                fill: false,
                tension: 0.2
            },
            {
                label: `${ciTest} - Error Band`,
                data: stats.map(s => ({ x: s.key, y: s.upper })),
                borderColor: chartColor(idx, 0),
                backgroundColor: chartColor(idx, 0.15),
                fill: '+1',
                pointRadius: 0,
                tension: 0.2
            },
            {
                label: `${ciTest} - Lower`,
                data: stats.map(s => ({ x: s.key, y: s.lower })),
                borderColor: chartColor(idx, 0),
                backgroundColor: chartColor(idx, 0.15),
                fill: false,
                pointRadius: 0,
                tension: 0.2
            }
        ];
    }).flat();

    if (chartPower) chartPower.destroy();
    chartPower = new Chart(document.getElementById('power-plot').getContext('2d'), {
        type: 'line',
        data: {
            datasets: powerDatasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            aspectRatio: 1.6,
            plugins: { 
                legend: { 
                    display: true,
                    filter: (legendItem) => !legendItem.text.includes('Band') && !legendItem.text.includes('Lower')
                } 
            },
            scales: {
                x: { 
                    title: { display: true, text: 'Effect Size' }, 
                    min: 0, 
                    max: 1,
                    type: 'linear'
                },
                y: { 
                    title: { display: true, text: 'Power' }, 
                    min: 0, 
                    max: 1 
                }
            }
        }
    });
}

function chartColor(idx, alpha = 1) {
    const colors = [
        `rgba(54, 162, 235, ${alpha})`,   // blue
        `rgba(255,99,132,${alpha})`,      // red
        `rgba(255, 205, 86, ${alpha})`,   // yellow
        `rgba(75, 192, 192, ${alpha})`,   // teal
        `rgba(153, 102, 255, ${alpha})`,  // purple
        `rgba(201, 203, 207, ${alpha})`,  // grey
        `rgba(0, 200, 83, ${alpha})`,     // green
        `rgba(255, 87, 34, ${alpha})`,    // deep orange
    ];
    return colors[idx % colors.length];
}