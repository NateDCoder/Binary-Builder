var inputs = [];
var outputs = [];

var currentIndex = 0;

for (let i = 0; i < 256; i++) {
    inputs.push(i);
    // outputs.push(Math.floor(10 * Math.sin(i)) + 10);
    outputs.push(i + 1);
}