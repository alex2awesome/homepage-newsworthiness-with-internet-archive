// custom scripts to load into Playwright. These specifically handle a heuristic page-segmentation
// technique that seeks to get the largest DOM areas.

get_parents = function (node) {
    `Given a child element in the DOM, return the path from ROOT to the child.`
    var nodes = [node]
    for (; node; node = node.parentNode) {
        nodes.unshift(node)
    }
    return nodes
}

function get_common_parent(node1, node2, return_common) {
    `Given two nodes in the DOM, return the common parent between them.
        * node1: the target node
        * node2: the comparator node
        * return_common (bool): 
            * if true, return the first node that they both share. 
            * if false, return the first node BEFORE the shared node, in the target node.
    `
    if (return_common === undefined)
        return_common = true

    var parents1 = get_parents(node1)
    var parents2 = get_parents(node2)

    if (parents1[0] !== parents2[0]){
        throw "No common ancestor!"
    }

    // parents are in order of top -> bottom
    for (var i = 0; i < parents1.length; i++) {
        if (parents1[i] !== parents2[i]){
            if (return_common)
                return parents1[i - 1]
            else
                return parents1[i]
        }
    }
}

function is_smaller_child(child_candidate, parent_candidate){
    `Given two nodes (in the same hierarchy), return:
        true if the "child_candidate" is a child of the "parent_candidate",
        false otherwise
    `
    var child_parents = get_parents(child_candidate)
    var parent_parents = get_parents(parent_candidate)
    return child_parents.length > parent_parents.length
}

function get_most_parent_with_joining(a_href, as, a_counts){
    `Method 1: Get the largest bounding box of link following 2 heuristics.
        1. If the link "a_href" only appears once, get the largest possible bounding box in the DOM hierarchy 
            that doesn't have any other links.
        2. If "a_href" appears multiple times in the DOM, get the smallest possible bounding box that covers both
            nodes. 
        
        params:
            * a_href: the target link that we wish to draw a box around
            * as: an Array of all DOM elements a
            * a_counts: a mapper from each href => [idx of nodes in as] containing that link.
    `
    var same_links = a_counts[a_href]
    if (same_links.length == 1){
        // If this is the only time this link appears, get the uppermost parent
        // that isn't also the parent of any other link.
        var i = same_links[0]
        var a = as[i]
        var curr_parent = document
        for (let j = 0; j < as.length; j++){
          if (i != j){
            var common_not_parent = get_common_parent(a, as[j], return_common=false)
            if (is_smaller_child(common_not_parent, curr_parent)){
              curr_parent = common_not_parent
            }
          }
        }
    } else {
        // Otherwise, get the greatest common parent for all the links.
        var all_instances_of_a = []
        a_counts[a_href].forEach(function(i){ all_instances_of_a.push(as[i]) })
        var curr_parent = all_instances_of_a[0]
        for (i=1; i < all_instances_of_a.length; i++){
            curr_parent = get_common_parent(curr_parent, all_instances_of_a[i])
        }
    }
    return curr_parent
}

function get_highest_singular_parent(i, as){
    `Similar to Method_1, we seek the largest possible bounding box. But here, we only follow one heuristic:
        1. Get the largest possible bounding box in the DOM hierarchy that doesn't have any other links.
        
        params:
            * i: the index of node "a" in "as
            * as: an Array of all DOM elements of type "A"
            * a_counts: a mapper from each href => [idx of nodes in as] containing that link.       
    `
    var a = as[i]
    var curr_parent = document
    for (let j = 0; j < as.length; j++){
      if ((i != j) & (as[i].href != as[j].href)) {
        var common_not_parent = get_common_parent(a, as[j], return_common=false)
        if (is_smaller_child(common_not_parent, curr_parent)){
          curr_parent = common_not_parent
        }
      }
    }
    return curr_parent
}

function get_text_of_node(node){
    `Get all the text associated with the node and it's children.
     Flexible in case there are non-textual children in the tree.`
    var iter = document.createNodeIterator(node, NodeFilter.SHOW_TEXT)
    var textnode;
    var output_text = ''

    // print all text nodes
    while (textnode = iter.nextNode()) {
      output_text = output_text + ' ' + textnode.textContent
    }
    return output_text
}

function bin_number_to_array(num, arr) {
    var curr = arr[0];
    var diff = Math.abs(num - curr);
    for (var val = 0; val < arr.length; val++) {
        var newdiff = Math.abs(num - arr[val]);
        if (newdiff < diff) {
            diff = newdiff;
            curr = arr[val];
        }
    }
    return curr;
}
