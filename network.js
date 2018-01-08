const _ = require('lodash')
const rvnorm = require('randgen').rvnorm

const e = Math.E;
const sigmoid = (x) => 1 / (1 + Math.pow(e, (x * -1)))
const sigmoidPrime = (x) => sigmoid(x) * (1 - sigmoid(x))

const generateZerosNetwork = (layerSizes) => ({
  biases: layerSizes.slice(1).map((size) => Array.apply(null, Array(size)).map(() => 0)),
  weights: _.zip(layerSizes.slice(0, -1), layerSizes.slice(1)).map((layer) =>
    Array.apply(null, Array(layer[1])).map(() => Array.apply(null, Array(layer[0])).map(() => 0))),
})

const generateNetwork = (layerSizes, weights, biases) => {
  return {
    numberOfLayers: layerSizes.length,
    layerSizes: layerSizes,
    biases: biases ? biases : layerSizes.slice(1).map((size) => rvnorm(size, 0, 1)),
    weights: weights ? weights : _.zip(layerSizes.slice(0, -1),
      layerSizes.slice(1)).map((layer) => {
        let layerWeights = [];
        for (let i = 0; i < layer[1]; i++) {
          layerWeights.push(rvnorm(layer[0], 0, 1))
        }
        return layerWeights
    }),
  }
}

const maxIndex = (array) => array.indexOf(Math.max.apply(null, array))

const evaluate = (network, data) =>
  data.reduce((sum, d) =>
    (maxIndex(feedForward(network, d.input)) === maxIndex(d.output)) ? sum + 1 : sum, 0)

const feedForward = (network, input) => _.zip(network.weights, network.biases)
  .map((layer) => _.zip(layer[0], layer[1]))
  .reduce((activation, layer) =>
    layer.map((neuron) =>
      sigmoid(neuron[0].reduce((sum, w, i) => sum + w * activation[i], 0) + neuron[1])), input)

const costDeriv = (output, expected) => output.map((n, i) => n - expected[i])

const backprop = (network, input, expected) => {
  const activations = [ input ]
  const zs = [];

  _.zip(network.weights, network.biases)
    .map((layer) => _.zip(layer[0], layer[1]))
    .reduce((activation, layer) => {
      const layerZ = []
      const layerActivation = layer.map((neuron) => {
        const z = neuron[0].reduce((sum, w, i) => sum + w * activation[i], 0) + neuron[1]
        layerZ.push(z)
        return sigmoid(z)
      })
      zs.push(layerZ)
      activations.push(layerActivation)
      return layerActivation
    }, input)

  const blank = generateZerosNetwork(network.layerSizes)
  const nablaBiases = blank.biases
  const nablaWeights = blank.weights

  let delta = costDeriv(activations[activations.length - 1], expected)
    .map((cost, i) => cost * sigmoidPrime(zs[zs.length - 1][i]))
  nablaBiases[nablaBiases.length - 1] = delta
  nablaWeights[nablaWeights.length - 1] = delta
    .map(d => activations[activations.length - 2].map(a => a * d))

  for (let i = 2; i < network.numberOfLayers; i++) {
    delta = network.weights[network.weights.length - i + 1]
      .reduce((sum, weights, j) => weights.map((w, k) => w * delta[j] + sum[k]))
      .map((n, j) => n * sigmoidPrime(zs[zs.length - i][j]))
    nablaBiases[nablaBiases.length - i] = delta
    nablaWeights[nablaWeights.length - i] = delta
      .map(d => activations[activations.length - i - 1].map(a => a * d))
  }

  return {
    biases: nablaBiases,
    weights: nablaWeights,
  }
}

const updateFromBatch = (network, batch, eta) => {
  const nabla = batch.reduce((grad, data) => {
    const delta = backprop(network, data.input, data.output)
    return {
      biases: delta.biases.map((layerBiases, i) =>
        layerBiases.map((bias, j) => bias + grad.biases[i][j])),
      weights: delta.weights.map((layerWeights, i) =>
        layerWeights.map((nWeights, j) =>
          nWeights.map((w, k) => w + grad.weights[i][j][k]))),
    }
  }, generateZerosNetwork(network.layerSizes))

  const newWeights = network.weights.map((layerW, i) =>
    layerW.map((neuronW, j) =>
      neuronW.map((w, k) => w - (eta / batch.length) * nabla.weights[i][j][k])))
  const newBiases = network.biases.map((layerB, i) =>
    layerB.map((bias, j) => bias - (eta / batch.length) * nabla.biases[i][j]))

  return generateNetwork(network.layerSizes, newWeights, newBiases)
}

const sgdRecur = (network, trainingData, totEpoch, epoch = 0, miniBatchSize, eta, testData) => {
  if (totEpoch - epoch - 1 === 0) {
    console.log('Stochastic Gradient Descent: Complete.')
    return network
  }
  const start = new Date()
  console.log(`Epoch ${epoch}: Starting`)
  const miniBatches = _.chunk(_.shuffle(trainingData), miniBatchSize)
  const newNetwork = miniBatches.reduce((net, batch) => {
    return updateFromBatch(net, batch, eta)
  }, network)

  if (testData)
    console.log(`Epoch ${epoch}: ${evaluate(newNetwork, testData)}/${testData.length}`)

  const end = new Date()
  const duration = (end.getTime() - start.getTime()) / 1000
  console.log(`Epoch ${epoch}: Complete.  Duration: ${duration}s`)
  return sgdRecur(newNetwork, trainingData, totEpoch, epoch + 1, miniBatchSize, eta, testData)
}

const stochGradDesc = (network, trainingData, numEpochs, miniBatchSize, eta, testData) => {
  return sgdRecur(network, trainingData, numEpochs, 0, miniBatchSize, eta, testData)
}

module.exports = {
  generateNetwork: generateNetwork,
  feedForward: feedForward,
  stochGradDesc: stochGradDesc,
  evaluate: evaluate,
  feedForward: feedForward,
  backprop, backprop,
}
