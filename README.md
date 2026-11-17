# Detective P-Hacker

This is a web game made for my Introduction to Data Science in Python (IDSE) project. You play as a corrupt newspaper editor trying to publish misleading data science headlines to get clicks.

## Concepts covered
* P-Hacking: The multiple comparisons problem. It shows how testing enough random variables will eventually give you a false positive (p < 0.05) just by pure chance.
* Correlation vs Causation: Seeing firsthand that a high Pearson's r doesn't prove a mechanism.
* Confounding Variables: Figuring out how a hidden third variable can drive a fake relationship, and seeing what happens when you control for it.

## How to run it locally
Make sure you have Python installed, then install the required libraries in your terminal:
```bash
pip install streamlit pandas numpy scipy
