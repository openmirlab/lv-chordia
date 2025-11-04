# Publishing lv-chordia to PyPI

This document provides instructions for publishing the lv-chordia package to PyPI.

## Two Publishing Methods

### Method 1: Automated Publishing with GitHub Actions (Recommended)

This repository includes a GitHub Actions workflow that automatically publishes to PyPI when you push a version tag.

### Method 2: Manual Publishing with Twine

Traditional manual upload using twine and API tokens.

---

## Method 1: Automated Publishing (GitHub Actions)

### Prerequisites

1. **GitHub Repository**: Push your code to GitHub
2. **PyPI Account**: Create an account at https://pypi.org/account/register/
3. **Trusted Publishing**: Set up PyPI Trusted Publishing (no API tokens needed!)

### Setup Trusted Publishing on PyPI

1. Go to https://pypi.org/manage/account/publishing/
2. Click "Add a new pending publisher"
3. Fill in the form:
   - **PyPI Project Name**: `lv-chordia`
   - **Owner**: Your GitHub username or organization
   - **Repository name**: `ISMIR2019-Large-Vocabulary-Chord-Recognition` (or your repo name)
   - **Workflow name**: `publish.yml`
   - **Environment name**: (leave blank)
4. Click "Add"

### Publishing Steps

```bash
# 1. Update version in pyproject.toml and __init__.py
# Already done: version = "1.0.0"

# 2. Commit your changes
git add .
git commit -m "Release version 1.0.0"

# 3. Create and push a version tag
git tag v1.0.0
git push origin main
git push origin v1.0.0

# 4. GitHub Actions will automatically:
#    - Build the package using UV
#    - Publish to PyPI using Trusted Publishing
#    - No API tokens needed!
```

### Monitor the Workflow

1. Go to your GitHub repository
2. Click on "Actions" tab
3. Watch the "Publish to PyPI" workflow run
4. If successful, your package will be on PyPI in a few minutes!

**Advantages:**
- ✅ No API tokens to manage
- ✅ More secure (Trusted Publishing)
- ✅ Automated on every version tag
- ✅ Built with UV for consistency
- ✅ No local configuration needed

---

## Method 2: Manual Publishing

### Prerequisites

1. **PyPI Account**: Create an account at https://pypi.org/account/register/
2. **API Token**: Generate an API token at https://pypi.org/manage/account/token/
3. **Build Tools**: Already installed via `uv add --dev build hatchling`

### Manual Publishing Steps

#### 1. Verify Package Quality

Before publishing, ensure the package is ready:

```bash
# Check that all files are committed
git status

# Verify the package builds successfully
uv build

# Check the built distributions
ls -lh dist/
```

#### 2. Test the Package Locally

```bash
# Install the package locally to test
uv pip install dist/lv_chordia-1.0.0-py3-none-any.whl

# Test the CLI
uv run lv-chordia --help

# Test the Python API
uv run python -c "import lv_chordia; print(lv_chordia.__version__)"
```

#### 3. Publish to Test PyPI (Recommended First)

Test PyPI allows you to practice the release process without affecting the production PyPI index.

```bash
# Add twine for uploading
uv add --dev twine

# Upload to Test PyPI
uv run twine upload --repository testpypi dist/lv_chordia-1.0.0*

# You'll be prompted for:
# Username: __token__
# Password: <your-testpypi-api-token>
```

**Test the installation from Test PyPI:**

```bash
# Create a test environment
uv venv test-env
source test-env/bin/activate  # On Windows: test-env\Scripts\activate

# Install from Test PyPI
pip install --index-url https://test.pypi.org/simple/ lv-chordia

# Test it works
python -c "import lv_chordia; print('Success!')"
```

#### 4. Publish to Production PyPI

Once you've verified everything works on Test PyPI:

```bash
# Upload to production PyPI
uv run twine upload dist/lv_chordia-1.0.0*

# You'll be prompted for:
# Username: __token__
# Password: <your-pypi-api-token>
```

#### 5. Verify the Published Package

```bash
# Wait a few minutes for PyPI to process the upload, then:
pip install lv-chordia

# Or with UV
uv add lv-chordia

# Test it works
python -c "import lv_chordia; print(lv_chordia.__version__)"
lv-chordia --help
```

## Using API Tokens

To avoid entering credentials each time, create a `~/.pypirc` file:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-YOUR_PRODUCTION_API_TOKEN_HERE

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-YOUR_TEST_API_TOKEN_HERE
```

**Important**: Make sure this file has restricted permissions:
```bash
chmod 600 ~/.pypirc
```

## Version Updates

For future releases:

1. Update version in `pyproject.toml`
2. Update version in `lv_chordia/__init__.py`
3. Update the README if needed
4. Commit changes
5. Create a git tag:
   ```bash
   git tag -a v1.0.1 -m "Release version 1.0.1"
   git push origin v1.0.1
   ```
6. Build and publish:
   ```bash
   # Clean old builds
   rm -rf dist/*

   # Build new version
   uv build

   # Upload to PyPI
   uv run twine upload dist/*
   ```

## Package Checklist

Before publishing, verify:

- [ ] Version number is correct in `pyproject.toml` and `__init__.py`
- [ ] README.md is comprehensive and accurate
- [ ] LICENSE file is present
- [ ] All authors are properly credited
- [ ] BibTeX citation is included for original research
- [ ] Package builds without errors (`uv build`)
- [ ] Package installs correctly locally
- [ ] CLI command works (`lv-chordia --help`)
- [ ] Python API works (`import lv_chordia`)
- [ ] Model files are included in the distribution
- [ ] Data files are included in the distribution
- [ ] All tests pass (if applicable)

## Package Contents Verification

Check what's included in your package:

```bash
# For wheel (.whl)
unzip -l dist/lv_chordia-1.0.0-py3-none-any.whl | head -50

# For source distribution (.tar.gz)
tar -tzf dist/lv_chordia-1.0.0.tar.gz | head -50
```

Verify that:
- All Python files are included
- Model files (*.sdict) are included
- Data files (*.txt) are included
- README.md and LICENSE are included
- No unnecessary files (.git, .venv, __pycache__) are included

## Distribution Size

Current package size:
- **Wheel**: ~777 KB (Python code and data files)
- **Source**: ~26 MB (includes model checkpoint files)

The large source distribution size is due to the 5 pre-trained model files (~5.7 MB each).

## Troubleshooting

### Package Too Large

If PyPI rejects the upload due to size:

1. **Option 1**: Host model files separately and download on first use
2. **Option 2**: Use Git LFS for model files
3. **Option 3**: Request size limit increase from PyPI

### Upload Fails

```bash
# Check your internet connection
# Verify your API token is correct
# Check that the version doesn't already exist on PyPI

# View detailed error:
uv run twine upload --verbose dist/lv_chordia-1.0.0*
```

### Wrong Files Included

Update `MANIFEST.in` and `pyproject.toml`:

```toml
# In pyproject.toml, adjust:
[tool.hatch.build.targets.wheel]
include = [...]

[tool.hatch.build.targets.sdist]
include = [...]
```

Then rebuild:
```bash
rm -rf dist/*
uv build
```

## Post-Publication

After publishing:

1. **Update GitHub Repository**:
   - Update README with PyPI badge
   - Create a GitHub release matching the version tag
   - Link to PyPI package page

2. **Announce**:
   - Post on relevant forums/communities
   - Update project website
   - Notify users of the new package

3. **Monitor**:
   - Watch for bug reports
   - Monitor download statistics
   - Respond to user issues

## PyPI Package Page

After publishing, your package will be available at:
- https://pypi.org/project/lv-chordia/

The page will show:
- Package description from README.md
- Installation instructions
- Project links
- Version history
- Download statistics

## Additional Resources

- **PyPI Help**: https://pypi.org/help/
- **Packaging Tutorial**: https://packaging.python.org/tutorials/packaging-projects/
- **Twine Documentation**: https://twine.readthedocs.io/
- **UV Documentation**: https://github.com/astral-sh/uv

---

**Good luck with your publication!**

For questions or issues, please open an issue on the GitHub repository.
