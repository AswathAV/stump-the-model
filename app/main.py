import streamlit as st
from PIL import Image

from matcher import search


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Stump the Model",
    page_icon="👟",
    layout="wide"
)


# =========================
# TITLE
# =========================

st.title("👟 Stump the Model")

st.write(
    "Upload a shoe image and find the most similar "
    "items from the 5,000-image catalogue."
)


# =========================
# UPLOAD IMAGE
# =========================

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


# =========================
# PROCESS IMAGE
# =========================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    st.subheader("Your Image")

    st.image(
        image,
        width=400
    )


    if st.button(
        "🔍 Find Matches"
    ):

        with st.spinner(
            "Searching catalogue..."
        ):

            results = search(
                image,
                top_k=5
            )


        st.subheader(
            "Top 5 Matches"
        )


        # Display results
        for rank, result in enumerate(
            results,
            start=1
        ):

            product_id = result[
                "product_id"
            ]

            similarity = result[
                "similarity"
            ]


            st.write(
                f"### {rank}. {product_id}"
            )

            st.write(
                f"Similarity: "
                f"{similarity:.4f}"
            )


            # Find catalogue image
            image_path = (
                f"data/catalogue/images/"
                f"{product_id}.jpg"
            )


            try:

                catalogue_image = Image.open(
                    image_path
                )

                st.image(
                    catalogue_image,
                    width=250
                )

            except:

                st.warning(
                    f"Could not find image "
                    f"for {product_id}"
                )


            st.divider()