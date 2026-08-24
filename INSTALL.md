# Using the Course Notebooks

The course uses Google Colab. You do not need Docker or a local Python installation for the labs.

## View the material

The easiest way to browse the lecture notebooks is through the static [course overview on nbviewer](https://nbviewer.org/github/coastalcph/nlp-course/blob/master/overview.ipynb).

## Work on a lab in Google Colab

1. Open the lab notebook from the [course schedule](README.md).
2. Select the **Open in Colab** link at the top of the notebook.
3. In Colab, select **File → Save a copy in Drive** before editing.
4. If the lab trains a neural model, select **Runtime → Change runtime type → GPU**.
5. Run the notebook from the beginning with **Runtime → Run all**.

Package-installation cells are included in each lab. Run them once at the beginning of a new Colab session. Colab sessions are temporary, so downloaded datasets and installed packages disappear when the runtime is reset.

## Save and submit your work

Your editable copy is stored in Google Drive. Use **File → Download → Download .ipynb** when you need a local copy or must include a notebook with a submission.

Do not edit the repository copy directly: course updates may replace it. Check the [course repository](https://github.com/coastalcph/nlp-course) and Absalon announcements for updates.

## Optional local use

Local execution is unsupported for the labs. If you nevertheless choose to work locally, create an isolated environment and install the packages used by the particular notebook. The teaching team may not be able to diagnose differences from the tested Colab environment.
