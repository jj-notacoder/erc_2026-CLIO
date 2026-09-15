# CLIO ERC bin geometry

The geometry verification and temporal tracker in
`erc_phase1_solution/bin_geometry.py`, the synthetic scene renderer in
`test/test_bin_geometry.py`, and the two RGB-D captures under
`test/data/bin_geometry/` are adapted from
[amanananah/clio_erc](https://github.com/amanananah/clio_erc), revision
`b9104920f486c7dbb54b2e8f726f2fcee04d3365`.

Original authors: CLIO ERC contributors. The source package declares the MIT
license in `src/clio_erc/package.xml`. Adaptations use this package's detection
types, colour masks, depth units, existing placement point convention, and
freshness/invalidation handling. No simulator model or controller changes were
imported.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
