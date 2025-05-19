class LogicLayer {
    constructor(prev_size, size, input_A_probs, input_B_probs, table_probs, index) {
        this.index = index;
        this.size = size;
        this.prev_size = prev_size;
        input_A_probs = input_A_probs
        input_B_probs = input_B_probs

        this.input_A_probs = softmax(input_A_probs, 1);
        this.input_B_probs = softmax(input_B_probs, 1);
        this.table_probs = softmax(table_probs, 0);


        const a_Zeroes = oneSubtractFromArray(addArrays(this.table_probs[0], this.table_probs[5], this.table_probs[10], this.table_probs[15]));

        const b_Zeroes = oneSubtractFromArray(addArrays(this.table_probs[0], this.table_probs[3], this.table_probs[12], this.table_probs[15]));
        let multiplier = subtract1FromArray(
            divArray(
                1,
                arrayClamp(matrixSum(matrixAdd(
                    arrayMatrixMutliplication(this.input_A_probs, a_Zeroes), 
                    arrayMatrixMutliplication(this.input_B_probs, b_Zeroes)), 0), 0, 1)
            )
        );
        console.log(multiplier)
        console.log("A Zeroes", a_Zeroes);
        console.log(this.table_probs)
        this.input_A_probs = softmax(
            arrayMatrixAddition(input_A_probs, multiplier),
            1
        );
        this.input_B_probs = softmax(
            arrayMatrixAddition(input_B_probs, multiplier),
            1
        );

        this.positions = [];
        this.startPositions = [];
        console.log(this.input_A_probs.length, this.input_A_probs[0].length);
        for (let i = 0; i < this.input_A_probs[0].length; i++) {
            if (index == 0) {
                this.startPositions.push(
                    createVector(number1.position.x + 30 / 2, number1.position.y + i * 30)
                );
            }
        }
        for (let i = 0; i < this.input_A_probs.length; i++) {
            this.positions.push(createVector(100 + index * 50, (i + 2) * 40));
        }

        this.selectedI = null;
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
        for (let i = 0; i < this.input_A_probs[0].length; i++) {
            for (let j = 0; j < this.input_A_probs.length; j++) {
                let startPositions =
                    this.index == 0
                        ? this.startPositions[i]
                        : nn.layers[this.index - 1].positions[i];
                if (!(this.table_probs[5][j] > 0.8) && !(this.table_probs[10][j] > 0.8)) {
                    stroke(
                        lerpColor(
                            color(100, this.input_A_probs[j][i] * 255),
                            color(255, 255, 0, this.input_A_probs[j][i] * 255),
                            transpose(this.prev_layer_output)[currentIndex][i]
                        )
                    );
                    line(
                        startPositions.x,
                        startPositions.y,
                        this.positions[j].x - 5,
                        this.positions[j].y - 5
                    );
                }
                // B Line
                if (!(this.table_probs[3][j] > 0.8) && !(this.table_probs[12][j] > 0.8)) {
                    stroke(
                        lerpColor(
                            color(100, this.input_B_probs[j][i] * 255),
                            color(255, 255, 0, this.input_B_probs[j][i] * 255),
                            transpose(this.prev_layer_output)[currentIndex][i]
                        )
                    );
                    line(
                        startPositions.x,
                        startPositions.y,
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

                text(
                    maxIndex + " " + int(this.table_probs[maxIndex][i] * 100),
                    this.positions[i].x + 25,
                    this.positions[i].y + 35
                );
            }
        }
    }
    update() {
        if (this.selectedI !== null) {
            this.positions[this.selectedI] = createVector(mouseX, mouseY);
        }
    }
    mousePressed(x, y) {
        for (let i = 0; i < this.positions.length; i++) {
            if (dist(x, y, this.positions[i].x, this.positions[i].y) < 15) {
                this.selectedI = i;
            }
        }
    }
    mouseReleased() {
        this.selectedI = null;
    }
}
