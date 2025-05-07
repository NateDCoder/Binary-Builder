var inputs = [];
var binaryInputs = [];

var outputs = [];
var binaryOutputs = [];

var currentIndex = 0;

for (let i = 0; i < 256; i++) {
    inputs.push(i);
    binaryInputs.push(dec2Bin(i).split('').map(Number))
    // outputs.push(Math.floor(10 * Math.sin(i)) + 10);
    outputs.push(i);
    binaryOutputs.push(dec2Bin(i).split('').map(Number))
}
