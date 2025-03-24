class ContextMenu {
  constructor(options, width = 120, height = 30) {
    this.options = options; // List of menu options
    this.width = width; // Width of the menu
    this.height = height; // Height of each menu option
    this.menuVisible = false;
    this.menuX = 0;
    this.menuY = 0;
  }

  // Show the menu at the given position
  show(x, y) {
    this.menuVisible = true;
    this.menuX = x;
    this.menuY = y;
  }

  // Hide the menu
  hide() {
    this.menuVisible = false;
  }

  // Draw the menu
  draw() {
    if (!this.menuVisible) return;
    fill(255);
    stroke(0);
    strokeWeight(2);
    rect(
      this.menuX,
      this.menuY,
      this.width,
      this.options.length * this.height,
      5
    );

    for (let i = 0; i < this.options.length; i++) {
      let yOffset = i * this.height;
      fill(240);
      rect(this.menuX, this.menuY + yOffset, this.width, this.height);
      fill(0);
      textSize(20);
      textAlign(LEFT, CENTER);
      text(
        this.options[i],
        this.menuX + 10,
        this.menuY + yOffset + this.height / 2
      );
    }
  }

  // Handle mouse click (detect option selection)
  handleClick(x, y) {
    if (!this.menuVisible) return;

    for (let i = 0; i < this.options.length; i++) {
      let yOffset = i * this.height;
      if (
        x > this.menuX &&
        x < this.menuX + this.width &&
        y > this.menuY + yOffset &&
        y < this.menuY + yOffset + this.height
      ) {
        this.hide();
        return { x: this.menuX, y: this.menuY, i: i };
      }
    }
    this.hide(); // Hide the menu after selection
    return;
  }
}
