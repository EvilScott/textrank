const sw = require('stopword'),
  { NGrams } = require('natural'),
  { Matrix, Vector } = require('vectorious');

Matrix.prototype.multiplyVec = function(vec) {
  return new Vector(this.multiply(new Matrix(vec, { shape: [ vec.length, 1 ] })));
};

Vector.prototype.l1Norm = function() {
  return this.reduce((sum, v) => sum + Math.abs(v), 0);
};

function cosineSimilarity(m, n) {
  return m.dot(n) / (m.magnitude() * n.magnitude());
}

function textrank(content, ngramLength = 8) {
  // sanitize content
  content = content
    .replace(/[?.!"'](?=\w)/g, '')
    .replace(/\W+/g, ' ')
    .trim()
    .toLowerCase();

  // extract vocab for vectorization
  const vocab = Array.from(new Set(sw.removeStopwords(content.split(' '))));

  // split into ngrams
  const phrases = NGrams.ngrams(content, ngramLength).map(n => n.join(' '));
  const oneHots = phrases.map(n => {
    return new Vector(vocab.map(w => n.indexOf(w) !== -1 ? 1 : 0));
  });

  // create graph
  let graph = Matrix.zeros(phrases.length, phrases.length);
  phrases.forEach((_, i) => {
    phrases.forEach((_, j) => {
      if (i === j) return;
      graph.set(i, j, cosineSimilarity(oneHots[i], oneHots[j]));
      if (i !== j) graph.set(j, i, cosineSimilarity(oneHots[i], oneHots[j]));
    })
  });

  // normalize graph
  graph = new Matrix(graph.T.toArray().map(row => {
    const sum = row.reduce((sum, x) => sum + x, 0);
    return row.map(v => v / sum);
  })).transpose();

  // do textrank
  const N = phrases.length;
  const eps = 0.0001;
  const d = 0.85;
  let v = Vector.random(N);
  v = v.scale(1 / v.l1Norm());
  let lastV = Vector.fill(N, 100);
  const mHat = graph.scale(d).add(Matrix.fill(N, N, (1 - d) / N));
  while (lastV.subtract(v).magnitude() > eps) {
    lastV = v;
    v = mHat.multiplyVec(v);
  }
  const ranks = v.toArray();

  // check for good data
  if (isNaN(ranks[0])) throw new Error('too many ngrams');

  // return top 5 ngrams
  return phrases.sort((a, b) => {
    return ranks[phrases.indexOf(b)] - ranks[phrases.indexOf(a)];
  }).slice(0, 5);
}
