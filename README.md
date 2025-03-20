# Maturaarbeit Céline Amélie Cattin

## Description

This project is part of a Maturaarbeit (final thesis) by Céline Amélie Cattin, created on November 4, 2024. It's a Python script that creates objects with two or three sides showing distinct images depending on the perspective of the viewer. 


## Installation

1. Clone the repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Configure the settings at the top of the script:
   - Select input images (MNIST index or file paths to png image)
   - Select output object size, color value thresholds to binarize the images , and optimization algorithm
   - Choose output options (STL file, wire model, construction plan for pearlmodel)
   - Choose additional output information (number of image points, minimal number of image points, number of possibilities to plot the optimum of the model) 

2. Run the script:
   ```
   python OneObjectTwoViews.py
   ```

3. Build or print the model :)

## License

MIT License 

## Acknowledgments

- Céline Amélie Cattin - Project creator

Citations:
[1] Céline Amélie Cattin, Ein Objekt, zwei Ansichten: Entwicklung eines Computerprogramms zur Visualisierung und Optimierung von Kunstwerken im Stil von Markus Raetz, Maturaarbeit Kantonsschule Wettingen, 2024
