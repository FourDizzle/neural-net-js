const fs = require('fs')
// const mnist = require('mnist')
const nn = require('./network')
var dataFileBuffer = fs.readFileSync(__dirname + '/../train-images-idx3-ubyte')
var labelFileBuffer = fs.readFileSync(__dirname + '/../train-labels-idx1-ubyte')
var trainingData = []

console.log('grabbing data...')
for (var image = 0; image <= 59999; image++) {
    var pixels = []

    for (var x = 0; x <= 27; x++) {
        for (var y = 0; y <= 27; y++) {
            pixels.push(dataFileBuffer[(image * 28 * 28) + (x + (y * 28)) + 16])
        }
    }
    var label = parseInt(JSON.stringify(labelFileBuffer[image + 8]), 10)
    var expectedOutput = [0,0,0,0,0,0,0,0,0,0]
    expectedOutput[label] = 1
    var imageData  = {
      input: pixels,
      output: expectedOutput,
    }

    trainingData.push(imageData);
}

console.log('making sets')
const trainingSet = trainingData.slice(0, 10000)
const testSet = trainingData.slice(10001, 11001)

const net = nn.generateNetwork([784, 30, 10])
console.log(nn.feedForward(net, trainingData[0].input))
nn.backprop(net, trainingData[0].input, trainingData[0].output)
// console.log(`Before training: ${nn.evaluate(net, testSet)}/${testSet.length}`)
// const trainedNet = nn.stochGradDesc(net, trainingSet, 30, 10, 3.0, testSet)
//
// console.log('Finished training, writing net to file!')
// fs.writeFile('./neural-net.json', JSON.stringify(trainedNet, null, 2), 'utf-8')
