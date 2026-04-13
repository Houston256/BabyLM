import html
import webbrowser
import os

from tokenizers import Tokenizer

def export_html_visualization(text: str, tokenizer: Tokenizer, output_file: str = "tokenizer_viz.html"):
    encoding = tokenizer.encode(text)

    html_str = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Tokenizer Visualization</title>
            <style>
                body {{
                    font-family: system-ui, -apple-system, sans-serif;
                    margin: 40px auto;
                    max-width: 900px;
                    color: #333;
                }}
                .token-container {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 8px;
                    padding: 20px;
                    background-color: #f9f9f9;
                    border: 1px solid #ddd;
                    border-radius: 8px;
                }}
                .token-box {{
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    padding: 6px 10px;
                    border-radius: 6px;
                    color: #000;
                    border: 1px solid rgba(0,0,0,0.1);
                    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
                }}
                .token-text {{
                    font-family: monospace;
                    font-size: 16px;
                    font-weight: bold;
                    white-space: pre-wrap; /* Ensures spaces aren't collapsed by HTML */
                }}
                .token-id {{
                    font-family: monospace;
                    font-size: 11px;
                    color: #444;
                    margin-top: 4px;
                    border-top: 1px solid rgba(0,0,0,0.1);
                    padding-top: 2px;
                    width: 100%;
                    text-align: center;
                }}
            </style>
        </head>
        <body>
            <h2>Tokenizer Visualization (with IDs)</h2>
            <div class="token-container">
        """

    for token, token_id in zip(encoding.tokens, encoding.ids):

        hue = (token_id * 137.5) % 360
        bg_color = f"hsl({hue}, 85%, 85%)"

        clean_token = token.replace("Ġ", " ")
        safe_token = html.escape(clean_token)

        if not clean_token.strip():
            safe_token = "&nbsp;" * len(clean_token)

        html_str += f"""
                <div class="token-box" style="background-color: {bg_color};">
                    <span class="token-text">{safe_token}</span>
                    <span class="token-id">{token_id}</span>
                </div>
            """

    html_str += f"""
            </div>
            <p><strong>Total Tokens:</strong> {len(encoding.tokens)}</p>
        </body>
        </html>
        """

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_str)

    print(f"Visualization saved to {output_file}")

    file_path = f"file://{os.path.abspath(output_file)}"
    webbrowser.open(file_path)