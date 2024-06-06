logistic = function(z){
    return  1 / (1 + Math.exp(-z))
}

function sum(arr){
    var total = 0;
    for (var i = 0; i < arr.length; i++) {
        total += arr[i];
    }
    return total
}

function _char_ngrams(text_document, ngram_range){
    `Tokenize text_document into a sequence of character n-grams`
    const text_len = text_document.length
    let min_n = ngram_range[0]
    const max_n = ngram_range[1]
    let n_grams = []
    if (min_n === 1){
        // no need to do any slicing for unigrams
        // iterate through the string
        n_grams = text_document.split('')
        min_n++
    }

    for (let n = min_n; n < Math.min(max_n + 1, text_len + 1); n++){
        for (let i = 0; i < text_len - n + 1; i++){
            n_grams.push(text_document.slice(i , i + n))
        }
    }
    return n_grams
}

var DOMAIN_BLACKLIST = [
    "google",
    "twitter",
    "facebook",
    "doubleclick",
    "eventbrite",
    "youtube",
    "vimeo",
    "instagram",
    "ceros"
]

var SUBDOMAIN_BLACKLIST = [
    "careers",
    "mail",
    "account",
    "events",
]

function get_valid_url(href) {
    `Constructs a URL class out of an href. Flexible in case the href is just the path.`
    try {
        try {
            return new URL(href);
        } catch (e) {
            return new URL(href, window.location.href)
        }
    } catch(e){
        return 'invalid URL'
    }
}

function is_banned_host(url){
    var host = url.hostname
    var domain = psl.parse(host).sld
    if (DOMAIN_BLACKLIST.indexOf(domain) != -1)
        return true

    var subdomain = psl.parse(host).subdomain
    if (SUBDOMAIN_BLACKLIST.indexOf(subdomain) != -1)
        return true

    return false
}

function clean_wayback_url_str(url_str){
    if (url_str.indexOf('https://web.archive.org') != -1 ){
        url_str = url_str.split('https://web.archive.org')[1]
        url_str = 'http' + url_str.split('http')[1]
    }
    return url_str
}

var LRUrlPredictor = class {
    constructor(model_obj, decision_threshold_prob) {
        // `decision_threshold_prob` is the decision-probability threshold for which we'll assign a class label.
        // i.e. "if it's greater than 50% probability, then it's a checkout page."
        this.threshold = decision_threshold_prob
        if (this.threshold == undefined){
            this.threshold = .5
        }

        this.ngram_range = model_obj.ngram_range
        this.lr_intercept_ = model_obj.lr_intercept[0]
        this.vocab = model_obj.vocab
        this.vector_length = this.vocab.length
        this.classes = model_obj.classes
        let class_to_idx = {}
        this.classes.forEach(function(d, i) { class_to_idx[d] = i })
        this.class_to_idx = class_to_idx
        this.lr_coef_ = model_obj.lr_coef
        if (this.lr_coef_.length == 1)
            this.lr_coef_ = {1 : this.lr_coef_[0]} // map True to the one columns of coefficients
    }

    get_vocab_vectors(n_grams){
        let that = this
        let sparse_idx = n_grams
            .map(function(d){ return that.vocab[d] })
            .filter(function(d){ return d !== undefined })
        return sparse_idx
    }

    get_prediction(url_str, class_label){
        if (class_label == undefined)
            class_label = true

        url_str = clean_wayback_url_str(url_str)
        var url = get_valid_url(url_str)
        if (url == 'invalid URL')
            return false
        if (is_banned_host(url))
            return false

        let clean_str = new URL(url).pathname
        let n_grams = _char_ngrams(clean_str, this.ngram_range)
        let vec = this.get_vocab_vectors(n_grams)
        var class_idx = this.class_to_idx[class_label]
        var coef = this.lr_coef_[class_idx]
        var active_coef = vec.map(function(word){ return coef[word]})
        var logits = sum(active_coef) + this.lr_intercept_
        var prob = logistic(logits)
        return prob > this.threshold
    }
}

var HueristicUrlPredictor = class {
    constructor(num_parts_threshold) {
        if (num_parts_threshold === undefined)
            num_parts_threshold = 5
        this.num_parts_threshold = num_parts_threshold
    }

    get_url_parts(url){
        `Get parts of the URL in case the site is not English`
        var path = url.pathname
        path = path.split(/[-/:.]/).filter(function(d){return d != ''})
        return path.length
    }

    get_prediction(url_str){
        url_str = clean_wayback_url_str(url_str)
        var url = get_valid_url(url_str)
        if (url == 'invalid URL')
            return false

        if (is_banned_host(url))
            return false

        return this.get_url_parts(url) > this.num_parts_threshold
    }
}


