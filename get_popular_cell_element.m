function [max_element] = get_popular_cell_element(list_of_element)
    [str, ~, lab] = unique(list_of_element(~cellfun('isempty',list_of_element)));
    cnt = sum(bsxfun(@eq, lab(:), 1:max(lab)));
    [max_val max_i] = max(cnt);
    max_element = str(find(cnt == max_val));
end