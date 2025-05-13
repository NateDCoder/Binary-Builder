class LogicLayer {
    constructor(prev_size, size, input_A_probs, input_B_probs, table_probs, index) {
        this.index = index;
        this.size = size;
        this.prev_size = prev_size;

        this.input_A_probs = softmax(input_A_probs, 1);
        this.input_B_probs = softmax(input_B_probs, 1);
        this.table_probs = softmax(table_probs, 0);

        let multiplier = subtract1FromArray(
            divArray(
                1,
                arrayClamp(matrixSum(matrixAdd(this.input_A_probs, this.input_B_probs), 0), 0, 1)
            )
        );
        console.log(input_A_probs.length, input_A_probs[0].length, multiplier.length);
        this.input_A_probs = softmax(arrayMatrixAddition(input_A_probs, multiplier), 1);
        this.input_B_probs = softmax(arrayMatrixAddition(input_B_probs, multiplier), 1);

        this.positions = [];
        this.startPositions = [];
        console.log(this.input_A_probs.length, this.input_A_probs[0].length);
        for (let i = 0; i < this.input_A_probs[0].length; i++) {
            if (index == 0) {
                this.startPositions.push(
                    createVector(number1.position.x + 30 / 2, number1.position.y + (i + 4) * 30)
                );
            } else {
                this.startPositions.push(createVector(150 + (index - 1) * 100, (i + 2) * 50));
            }
        }
        for (let i = 0; i < this.input_A_probs.length; i++) {
            this.positions.push(createVector(150 + index * 100, (i + 2) * 50));
        }
    }

    forward(prev_layer_output) {
        this.prev_layer_output = prev_layer_output;
        this.a_inputs = matmul(this.input_A_probs, prev_layer_output);
        this.b_inputs = matmul(this.input_B_probs, prev_layer_output);

        let table_output = tableOpperator(this.a_inputs, this.b_inputs);

        this.result = weightedLogicOutput(table_output, this.table_probs);
        return this.result;
    }

    show() {
        console.log(
            this.input_A_probs[0].length,
            this.input_A_probs.length,
            this.startPositions.length,
            this.positions.length
        );
        for (let i = 0; i < this.input_A_probs[0].length; i++) {
            for (let j = 0; j < this.input_A_probs.length; j++) {
                if (!(this.table_probs[5][j] > 0.8)) {
                    stroke(
                        lerpColor(
                            color(100, this.input_A_probs[j][i] * 255),
                            color(255, 255, 0, this.input_A_probs[j][i] * 255),
                            transpose(this.prev_layer_output)[currentIndex][i]
                        )
                    );
                    line(
                        this.startPositions[i].x,
                        this.startPositions[i].y,
                        this.positions[j].x - 5,
                        this.positions[j].y - 5
                    );
                }
                // B Line
                if (!(this.table_probs[3][j] > 0.8)) {
                    stroke(
                        lerpColor(
                            color(100, this.input_B_probs[j][i] * 255),
                            color(255, 255, 0, this.input_B_probs[j][i] * 255),
                            transpose(this.prev_layer_output)[currentIndex][i]
                        )
                    );
                    line(
                        this.startPositions[i].x,
                        this.startPositions[i].y,
                        this.positions[j].x - 5,
                        this.positions[j].y + 5
                    );
                }
            }
        }
        for (let j = 0; j < this.input_A_probs.length; j++) {
            fill(
                lerpColor(color(100), color(255, 255, 0), transpose(this.result)[currentIndex][j])
            );
            noStroke();
            circle(this.positions[j].x, this.positions[j].y, 15);
        }
        // console.log(transpose(this.a_inputs)[currentIndex]);
        // console.log(transpose(this.result)[currentIndex])
        // console.log(this.input_A_probs[currentIndex]);
        // console.log(number1.binary);

        // stroke(lerpColor(color(100), color(255, 255, 0), this.supplierA()));
        // line(this.aPosition.x, this.aPosition.y, this.position.x, this.position.y);

        // stroke(lerpColor(color(100), color(255, 255, 0), this.supplierB()));
        // line(this.bPosition.x, this.bPosition.y, this.position.x, this.position.y);
    }

    showGUI() {
        noStroke();
        fill(180);
        for (let i = 0; i < this.positions.length; i++) {
            if (dist(mouseX, mouseY, this.positions[i].x, this.positions[i].y) < 15) {
                rect(this.positions[i].x, this.positions[i].y, 50, 50);
                fill(0);
                text(
                    int(100 * transpose(this.result)[currentIndex][i]) / 100,
                    this.positions[i].x + 25,
                    this.positions[i].y + 15
                );

                const maxIndex = transpose(this.table_probs)[i].reduce(
                    (maxIdx, curr, idx, arr) => (curr > arr[maxIdx] ? idx : maxIdx),
                    0
                );

                text(maxIndex, this.positions[i].x + 25, this.positions[i].y + 35);
            }
        }
    }
}
