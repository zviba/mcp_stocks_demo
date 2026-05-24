import streamlit as st
from openai import OpenAI

# Page configuration
st.set_page_config(page_title="Story Generator", page_icon="📖")

# App title
st.title("📖 Story Generator")
st.write("Enter a person's name and I'll create a creative story about them!")

# OpenAI API Key input
st.subheader("Configuration")
openai_api_key = st.text_input(
    "OpenAI API Key",
    value=st.session_state.get("openai_api_key", ""),
    type="password",
    help="Enter your OpenAI API key to enable story generation",
    key="openai_api_key_input"
)
if openai_api_key:
    st.session_state["openai_api_key"] = openai_api_key

# Person name input
st.subheader("Story Input")
person_name = st.text_input(
    "Person's Name",
    placeholder="Enter the name of the person...",
    key="person_name_input"
)

# Story style selection
story_style = st.selectbox(
    "Story Style",
    options=["Adventure", "Fantasy", "Mystery", "Sci-Fi", "Romance", "Comedy", "Drama", "General"],
    index=7,
    help="Choose the style/genre for the story"
)

# Generate button
generate_button = st.button("Generate Story", type="primary")

# Generate and display story
if generate_button:
    if not person_name:
        st.error("Please enter a person's name!")
    elif not openai_api_key:
        st.error("Please enter your OpenAI API key!")
    else:
        with st.spinner("Generating your story..."):
            try:
                # Initialize OpenAI client
                client = OpenAI(api_key=openai_api_key)
                
                # Create the prompt
                prompt = f"""Write a creative and engaging story about a person named {person_name}. 
The story should be in a {story_style.lower()} style. Make it interesting, well-written, and approximately 300-500 words long. 
The story should be fictional but feel realistic and compelling."""
                
                # Generate story using OpenAI
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a creative storyteller who writes engaging and imaginative stories."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1000,
                    temperature=0.8
                )
                
                story = response.choices[0].message.content
                
                # Display the story
                st.subheader(f"Story about {person_name}")
                st.write("---")
                st.write(story)
                st.write("---")
                
                # Success message
                st.success("Story generated successfully!")
                
            except Exception as e:
                st.error(f"Error generating story: {str(e)}")
                st.info("Please check your API key and try again.") 