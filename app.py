import anthropic

client = anthropic.Anthropic(
    # defaults to os.environ.get("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")
    api_key="my_api_key",
)

message = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1000,
    temperature=0.0,
    system="Respond only in Yoda-speak.",
    messages=[
        {"role": "user", "content": "How are you today?"}
    ]
)

print(message.content)
