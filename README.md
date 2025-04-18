# AI Blog Writing Agent

This Python script automates the process of generating SEO-optimized blog posts using AI. It leverages Google Gemini for content generation and analysis, and external APIs for research.

## Features

*   **Topic Analysis:** Analyzes a given topic to determine subtopics, target audience, technical depth, and relevant keywords using [`TopicAnalysisAgent`](main.py).
*   **Research Gathering:** Fetches recent news articles (NewsData.io), related keywords (Datamuse API), and relevant quotes (Quotable.io) using [`ResearchAgent`](main.py).
*   **Content Generation:** Creates a full blog post including title, introduction, sections for each subtopic, and conclusion using [`ContentGenerationAgent`](main.py).
*   **SEO Optimization:** Generates meta descriptions, URL slugs, and SEO tags for the blog post using [`SEOOptimizationAgent`](main.py).
*   **Export:** Saves the generated blog post as a Markdown file and associated metadata as a JSON file in the `output/` directory using [`ExportAgent`](main.py).
*   **Batch Processing:** Supports generating multiple blog posts from a list of topics.
*   **Readability Score:** Calculates the Flesch-Kincaid Grade Level for the generated content.

## Requirements

*   Python 3.7+
*   Required Python packages:
    *   `google-generativeai`
    *   `python-dotenv`
    *   `aiohttp`
    *   *(Note: The script includes a custom `readability` class for Flesch-Kincaid calculation)*

## Setup

1.  **Clone the repository (if applicable) or ensure you have the `main.py` file.**
2.  **Install dependencies:**
    ```bash
    pip install google-generativeai python-dotenv aiohttp
    ```
3.  **Create a `.env` file** in the root directory with your API keys:
    ```dotenv
    # .env
    GEMINI_API_KEY=YOUR_GOOGLE_GEMINI_API_KEY
    NEWSDATA_API_KEY=YOUR_NEWSDATA_IO_API_KEY
    ```
    *   Get your Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
    *   Get your NewsData.io API key from [NewsData.io](https://newsdata.io/). (Optional, mock data is used if the key is missing).
4.  **Create an output directory:**
    ```bash
    mkdir output
    ```
    *(Note: The script creates this directory if it doesn't exist, but it's good practice. Ensure  is included in your  if using version control, as provided in the example .gitignore.)*

## Usage

Run the script from your terminal using the following command-line arguments:

**Generate a single blog post:**

```bash
python main.py --topic "Your Blog Post Topic" --tone "desired tone" --output "output_directory"

--topic: (Required) The main topic for the blog post.
--tone: (Optional) The desired tone (e.g., "educational", "formal", "creative"). Defaults to "educational".
--output: (Optional) The directory to save output files. Defaults to "output".

Output
The script generates the following files in the specified output directory (default: output):

Markdown File (.md): Contains the full blog post content with YAML front matter for metadata (title, description, slug, tags, reading time).
Filename format: {slug}-{timestamp}.md
Metadata File (.json): Contains detailed metadata about the generated blog post, including title, SEO details, topic analysis parameters, word count, reading time, and creation timestamp.
Filename format: {slug}-{timestamp}-metadata.json
The script will print a summary to the console upon completion, including file paths and basic statistics.