# ucsdml.github.io

To set up
```
gem install bundler
bundle install
```

Everytime you update something, execute
```
bundle exec jekyll serve
```

Then copy the contents in the `_site` directory to the master branch to update the blog.

### Style of blog posts

Currently, you can choose from the following blog post layouts:

- `post` (default): a standard blog post
- `distill`: a distill publication-like post (see [distill.pub](https://distill.pub/))

Choose the desired layout by adding the following to the YAML front matter of your markdown file for your post:

``` 
layout: <one-of-the-above>
```