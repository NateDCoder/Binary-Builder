function dec2Bin(dec) {
  if (dec >= 256) {
    dec = 255; // Cap at 255 if the number exceeds 8-bit limit
  }
  if (dec >= 0) {
    return dec.toString(2).padStart(8, "0"); // Add leading zeros for 8-bit binary
  } else {
    let binary = ((~-dec + 1) >>> 0).toString(2); // Handle negative numbers with 2's complement
    return binary.slice(-8).padStart(8, "0"); // Ensure it's 8 bits
  }
}

function bin2Dec(bin) {
  if (bin.length !== 8) {
    console.error("Binary string must be 8 bits long.");
    return null;
  }

  // Handle negative 2's complement conversion

  return parseInt(bin, 2);
}

function isPointInRect(px, py, rx, ry, rw, rh) {
  return px >= rx && px <= rx + rw && py >= ry && py <= ry + rh;
}

function replaceBit(bin, position, newBit) {
  if (position < 0 || position >= bin.length) {
    console.error(
      "Invalid position. Must be between 0 and " + (bin.length - 1)
    );
    return bin;
  }
  if (newBit !== "0" && newBit !== "1") {
    console.error("Invalid bit. Must be '0' or '1'.");
    return bin;
  }

  // Replace the bit at the specified position and return the modified string
  return bin.substring(0, position) + newBit + bin.substring(position + 1);
}

function seededRandom(seed) {
  let state = seed % 2147483647;
  if (state <= 0) state += 2147483646;

  return function () {
    state = (state * 16807) % 2147483647;
    return (state - 1) / 2147483646;
  };
}

const rng = seededRandom(12345); // Seed value


function softmax(arr) {
  // Step 1: Calculate exponentials and handle large numbers
  const maxVal = Math.max(...arr); // Prevent large exponent values
  const exps = arr.map(x => Math.exp(x - maxVal));

  // Step 2: Sum of exponentials
  const sumExps = exps.reduce((sum, val) => sum + val, 0);

  // Step 3: Divide each exponential by the sum to get probabilities
  return exps.map(val => val / sumExps);
}