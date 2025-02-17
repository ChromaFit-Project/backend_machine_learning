# Skin Tone-Based Clothing Color Recommendation

## Overview
This project aims to classify skin tones and recommend suitable clothing colors based on their HEX codes. It utilizes machine learning techniques to categorize skin shades and map them to predefined color palettes, helping users choose clothing colors that complement their skin tone. This can be beneficial for fashion stylists, online clothing retailers, and individuals looking to enhance their wardrobe choices.

## Project Structure
- `skin_shades_india.csv` - Contains 56 unique skin shades with HEX codes, sourced from a diverse set of skin tones.
- `fashion_color_palette.csv` - Includes categorized clothing colors with HEX codes, grouped based on their suitability for different skin tones.
- `skin_tone_mapping.py` - The main script that processes data, classifies skin tones, and maps them to recommended clothing colors.
- `final_recommendations.csv` - The output file containing mapped skin tones and recommended clothing colors.
- `README.md` - This document, providing an overview of the project.

## Methodology
1. **Data Preprocessing**
   - Extract HEX codes from datasets.
   - Convert HEX codes to RGB values for numerical processing.
   - Normalize and clean the dataset to remove inconsistencies.
2. **Skin Tone Classification**
   - Use KMeans clustering to classify skin tones into predefined categories.
   - Map each skin shade to the closest category from `fashion_color_palette.csv`.
   - Ensure balanced categorization by analyzing cluster distributions.
3. **Color Recommendation**
   - Assign suitable clothing colors based on skin tone classification.
   - Merge datasets to generate final recommendations.
   - Provide a ranked list of colors with suitability scores based on clustering results.

## Dependencies
- Python 3.x
- Pandas - For handling structured data.
- NumPy - For numerical computations.
- Scikit-learn - For clustering and classification.
- Matplotlib - For visualizing skin tone distributions (optional).

## How to Run
1. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib
   ```
2. Run the script:
   ```bash
   python skin_tone_mapping.py
   ```
3. The output file `final_recommendations.csv` will contain:
   - Skin Shade ID
   - Skin HEX Code
   - Assigned Skin Tone Category
   - Recommended Clothing Colors
   - Clothing Color HEX Codes

## Example Output
| Skin Shade ID | Skin HEX Code | Assigned Category | Recommended Clothing Colors | Clothing Color HEX |
|--------------|--------------|-------------------|---------------------------|-------------------|
| Shade 1 | #F5E0D8 | Cool Tones | Navy Blue, Emerald Green | #000080, #50C878 |
| Shade 2 | #C68642 | Warm Tones | Coral, Mustard Yellow | #FF7F50, #FFDB58 |

## Future Improvements
- Enhance clustering accuracy using deep learning models like CNNs.
- Expand dataset with more diverse skin tones from global datasets.
- Develop a web or mobile application for real-time recommendations.
- Implement an API for easy integration with e-commerce platforms.
- Add user feedback loops to refine recommendations based on preferences.
