import openai
import os

# Set your OpenAI API key here
openai.api_key = os.getenv("OPENAI_API_KEY")

PROMPT = "Explain quantum computing to a 10-year-old"

def query_gpt(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error: {e}"

# If you don’t have the Claude API, paste the response manually
def query_claude(prompt):
    return """Imagine your regular computer is like a coin that can only show heads OR tails - it has to pick one. That's how normal computers work with 1s and 0s.
              But a quantum computer is like a magical spinning coin that can be heads AND tails at the same time! This is called "superposition" - it's like the coin is spinning so fast you can see both sides.
              Here's why this is super cool:
              Regular computers solve problems step by step:

              Try door #1, then door #2, then door #3...
              Like checking every book in a library one by one

              Quantum computers can try many solutions at once:

              It's like having a magic helper who can walk through ALL the doors at the same time
              Or checking every book in the library simultaneously!

              What makes it tricky:

              These quantum computers have to be kept colder than outer space (brrr!)
              They're very delicate - even tiny vibrations can mess them up
              When you try to peek at the answer, the "spinning coin" stops and picks just one side

              What could they do?

              Help doctors create new medicines much faster
              Solve traffic problems for entire cities
              Keep our internet passwords super safe

              Right now, quantum computers are like the very first cars - big, expensive, and only experts can use them. But one day, they might help solve problems that would take regular computers thousands of years!
              It's like having a crystal ball that can see all possible answers at once, then picks the best one."""

def create_markdown(prompt, gpt_response, claude_response):
    with open("llm_comparison.md", "w", encoding="utf-8") as f:
        f.write("# LLM Comparison Report\n\n")
        f.write(f"## Prompt:\n{prompt}\n\n")
        f.write("## GPT-3.5 Response:\n")
        f.write(gpt_response + "\n\n")
        f.write("## Claude Response:\n")
        f.write(claude_response + "\n\n")
        f.write("## Comparison:\n")
        f.write("- GPT used... (write your observations here)\n")
        f.write("- Claude used... (write your observations here)\n")

if __name__ == "__main__":
    gpt_output = query_gpt(PROMPT)
    claude_output = query_claude(PROMPT)

    print("=== GPT-3.5 ===\n", gpt_output)
    print("\n=== Claude ===\n", claude_output)

    create_markdown(PROMPT, gpt_output, claude_output)
    print("\n✅ Report saved to llm_comparison.md")
