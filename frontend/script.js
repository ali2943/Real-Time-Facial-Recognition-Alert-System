const queryEl = document.getElementById('query');
const runBtn = document.getElementById('runBtn');
const clearBtn = document.getElementById('clearBtn');
const errorEl = document.getElementById('error');

const phaseTargets = {
  output: document.getElementById('output'),
  symbol_table: document.getElementById('symbol_table'),
  semantic: document.getElementById('semantic'),
  intermediate_code: document.getElementById('intermediate_code'),
  optimized_code: document.getElementById('optimized_code'),
  code_generation: document.getElementById('code_generation')
};

const lexicalContainer = document.getElementById('lexical');

function setError(message = '') {
  if (!message) {
    errorEl.classList.add('hidden');
    errorEl.textContent = '';
    return;
  }
  errorEl.textContent = message;
  errorEl.classList.remove('hidden');
}

function renderTokens(tokens = []) {
  lexicalContainer.innerHTML = '';
  tokens.forEach((token) => {
    const span = document.createElement('span');
    span.className = `token ${token.category || ''}`;
    span.textContent = `${token.type}(${token.value})`;
    lexicalContainer.appendChild(span);
  });
}

function stringify(value) {
  if (typeof value === 'string') return value;
  return JSON.stringify(value, null, 2);
}

function clearOutputs() {
  Object.values(phaseTargets).forEach((el) => {
    el.textContent = '';
  });
  renderTokens([]);
  setError();
}

async function runQuery() {
  const query = queryEl.value.trim();
  if (!query) {
    setError('Please enter a SQL query before running.');
    return;
  }

  setError();
  runBtn.disabled = true;
  runBtn.textContent = 'Running...';

  try {
    const response = await fetch('/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query })
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || 'Query execution failed');
    }

    renderTokens(data.tokens || []);
    Object.entries(phaseTargets).forEach(([key, target]) => {
      target.textContent = stringify(data[key]);
    });
  } catch (error) {
    setError(error.message || 'Unexpected error');
  } finally {
    runBtn.disabled = false;
    runBtn.textContent = '▶ Run';
  }
}

runBtn.addEventListener('click', runQuery);

clearBtn.addEventListener('click', () => {
  queryEl.value = '';
  clearOutputs();
});

document.querySelectorAll('.sample').forEach((btn) => {
  btn.addEventListener('click', () => {
    queryEl.value = btn.dataset.query || '';
    setError();
  });
});

queryEl.value = "CREATE TABLE students (id, name, age);";
