function [state_action_feats, prev_grid, prev_head_loc] = extract_state_action_features(prev_grid, grid, prev_head_loc, nbr_feats)
%
% Code may be changed in this function, but only where it states that it is 
% allowed to do so.
%
% Function to extract state-action features, based on current and previous
% grids (game screens).
%
% Input:
%
% prev_grid     - Previous grid (game screen), N-by-N matrix. If initial
%                 time-step: prev_grid = grid, else: prev_grid != grid.
% grid          - Current grid (game screen), N-by-N matrix. If initial
%                 time-step: prev_grid = grid, else: prev_grid != grid.
% prev_head_loc - The previous location of the head of the snake (from the 
%                 previous time-step). If initial time-step: Assumed known,
%                 else: inferred in function "update_snake_grid.m" (so in
%                 practice it will always be known in this function).
% nbr_feats     - Number of state-action features per action. Set this 
%                 value appropriately in the calling script "snake.m", to
%                 match the number of state-action features per action you
%                 end up using.
%
% Output:
%
% state_action_feats - nbr_feats-by-|A| matrix, where |A| = number of
%                      possible actions (|A| = 3 in Snake), and nbr_feats
%                      is described under "Input" above. This matrix
%                      represents the state-action features extracted given
%                      the current and previous grids (game screens).
% prev_grid          - The previous grid as seen from one step in the
%                      future, i.e., prev_grid is set to the input grid.
% prev_head_loc      - The previous head location as seen from one step
%                      in the future, i.e., prev_head_loc is set to the
%                      current head location (the current head location is
%                      inferred in the code below).
%
% Bugs, ideas etcetera: send them to the course email.

% --------- DO NOT CHANGE ANYTHING BELOW UNLESS OTHERWISE NOTIFIED! -------

% Extract grid size.
N = size(grid, 1);

% Initialize state_action_feats to nbr_feats-by-3 matrix.
state_action_feats = nan(nbr_feats, 3);

% Based on how grid looks now and at previous time step, infer head
% location.
change_grid = grid - prev_grid;
prev_grid   = grid; % Used in later calls to "extract_state_action_features.m"

% Find head location (initially known that it is in center of grid).
if nnz(change_grid) > 0 % True, except in initial time-step
    [head_loc_m, head_loc_n] = find(change_grid > 0);
else % True only in initial time-step
    head_loc_m = round(N / 2);
    head_loc_n = round(N / 2);
end
head_loc = [head_loc_m, head_loc_n];

% Previous head location.
prev_head_loc_m = prev_head_loc(1);
prev_head_loc_n = prev_head_loc(2);

% Infer current movement directory (N/E/S/W) by looking at how current and previous
% head locations are related
if prev_head_loc_m == head_loc_m + 1 && prev_head_loc_n == head_loc_n     % NORTH
    movement_dir = 1;
elseif prev_head_loc_m == head_loc_m && prev_head_loc_n == head_loc_n - 1 % EAST
    movement_dir = 2;
elseif prev_head_loc_m == head_loc_m - 1 && prev_head_loc_n == head_loc_n % SOUTH
    movement_dir = 3;
else                                                                      % WEST
    movement_dir = 4;
end

% The current head_loc will at the next time-step be prev_head_loc.
prev_head_loc = head_loc;

% ------------- YOU MAY CHANGE SETTINGS BELOW! --------------------------

% HERE BEGINS YOUR STATE-ACTION FEATURE ENGINEERING. THE CODE BELOW IS 
% ALLOWED TO BE CHANGED IN ACCORDANCE WITH YOUR CHOSEN FEATURES. 
% Some skeleton code is provided to help you get started. Also, have a 
% look at the function "get_next_info" (see bottom of this function).
% You may find it useful.
[apple_loc_m,apple_loc_n] = find(grid ==-1);
apple_loc = [apple_loc_m, apple_loc_n];
% Find head location (initially known that it is in center of grid).
if nnz(change_grid) > 0 % True, except in initial time-step
    [tail_loc_m, tail_loc_n] = find(change_grid < 0);
else % True only in initial time-step
    tail_loc_m = round(N / 2)+9;
    tail_loc_n = round(N / 2);
end
tail_loc = [tail_loc_m, tail_loc_n];

snake_length = sum(sum(grid)) + 1 - 116;



for action = 1 : 3 % Evaluate all the different actions (left, forward, right).
    
    % Feel free to uncomment below line of code if you find it useful.
    [next_head_loc, next_move_dir] = get_next_info(action, movement_dir, head_loc);
    
    hit_the_wall=0;
   % hit_the_wall = grid(next_head_loc(1), next_head_loc(2));
     if (grid(next_head_loc(1), next_head_loc(2))==1)
         hit_the_wall =1;
     end
    
%     c = 0;
%     m = next_head_loc(1);
%     n = next_head_loc(2);
%     
%     while (grid(m, n) ~= 1)
%         c = c+1;
%         switch (next_move_dir)
%             case (1) % NORTH
%                 m = m -1;
%             case (2) % EAST
%                 n = n + 1;
%             case (3) % SOUTH
%                 m = m + 1;
%             case (4) % WEST
%                 n = n - 1;
%         end
%     end
%     wall_dist_inv = 1.0 / (c+2);
%     
    
    trapped = 0;
    if (hit_the_wall ~= 1)
        [tmpGrid, nSpace] = availableSpace(grid, next_head_loc(1), next_head_loc(2), 0);      
        if (nSpace < snake_length)
            trapped = 1;
        end
    end
    
    
    
%     next_to_wall = 0;
%     if (hit_the_wall ~= 1)
%         temp = (grid(next_head_loc(1) + 1, next_head_loc(2)) + ...
%             grid(next_head_loc(1), next_head_loc(2) + 1)+...
%             grid(next_head_loc(1) - 1, next_head_loc(2)) +...
%             grid(next_head_loc(1), next_head_loc(2) - 1));
%         if (temp > 1)
%             next_to_wall = 1;
%         end
%     end

%dist_to_wall = min([next_head_loc(1) - 1, N - next_head_loc(1), next_head_loc(2) - 1, N - next_head_loc(2)]) / N;    
%     if (dist_to_wall == 0)
%         hit_the_wall = 1;
%     end
    % Replace this to fit the number of state-action features per features
    % you choose (3 are used below), and of course replace the randn() 
    % by something more sensible.
    
%      state_action_feats(1, action) = sum((next_head_loc-apple_loc).^2)/2/N^2;
%      state_action_feats(2, action) = sum((next_head_loc-tail_loc).^2)/2/N^2;
%      state_action_feats(3, action) = hit_the_wall;

    

%     state_action_feats(1, action) = sum((next_head_loc-apple_loc).^2)/2/N^2;
%     state_action_feats(2, action) = trapped; %0;%sum((next_head_loc-tail_loc).^2)/2/N^2;
%     state_action_feats(3, action) = hit_the_wall;
% 
    state_action_feats(1, action) = abs(next_head_loc(1)-apple_loc(1)) / N;
    state_action_feats(2, action) = abs(next_head_loc(2)-apple_loc(2)) / N;
    state_action_feats(3, action) = hit_the_wall;
    state_action_feats(4, action) = trapped;
%     
    % ... and so on ...
end
end

% To get the trapped white spaces in front of the snake
function [newGrid, newN] = availableSpace(grid, x, y, n)
    grid(x, y) = 2;
    n = n+1;
    if (grid(x-1, y) == 0)
        [grid, n] = availableSpace(grid, x-1, y, n);
    end
    if (grid(x+1, y) == 0)
        [grid, n] = availableSpace(grid, x+1, y, n);
    end
    if (grid(x, y-1) == 0)
        [grid, n] = availableSpace(grid, x, y-1, n);
    end
    if (grid(x, y+1) == 0)
        [grid, n] = availableSpace(grid, x, y+1, n);
    end
    newGrid = grid;
    newN = n;
end


%
% DO NOT CHANGE ANYTHING IN THE FUNCTION get_next_info BELOW!
%
function [next_head_loc, next_move_dir] = get_next_info(action, movement_dir, head_loc)
% Function to infer next haed location and movement direction

% Extract relevant stuff
head_loc_m = head_loc(1);
head_loc_n = head_loc(2);

if movement_dir == 1 % NORTH
    if action == 1     % left
        next_head_loc_m = head_loc_m;
        next_head_loc_n = head_loc_n - 1;
        next_move_dir   = 4; 
    elseif action == 2 % forward
        next_head_loc_m = head_loc_m - 1;
        next_head_loc_n = head_loc_n;
        next_move_dir   = 1;
    else               % right
        next_head_loc_m = head_loc_m;
        next_head_loc_n = head_loc_n + 1;
        next_move_dir   = 2;
    end
elseif movement_dir == 2 % EAST
    if action == 1
        next_head_loc_m = head_loc_m - 1;
        next_head_loc_n = head_loc_n;
        next_move_dir   = 1;
    elseif action == 2
        next_head_loc_m = head_loc_m;
        next_head_loc_n = head_loc_n + 1;
        next_move_dir   = 2;
    else
        next_head_loc_m = head_loc_m + 1;
        next_head_loc_n = head_loc_n;
        next_move_dir   = 3;
    end
elseif movement_dir == 3 % SOUTH
    if action == 1
        next_head_loc_m = head_loc_m;
        next_head_loc_n = head_loc_n + 1;
        next_move_dir   = 2;
    elseif action == 2
        next_head_loc_m = head_loc_m + 1;
        next_head_loc_n = head_loc_n;
        next_move_dir   = 3;
    else
        next_head_loc_m = head_loc_m;
        next_head_loc_n = head_loc_n - 1;
        next_move_dir   = 4;
    end
else % WEST
    if action == 1
        next_head_loc_m = head_loc_m + 1;
        next_head_loc_n = head_loc_n;
        next_move_dir   = 3;
    elseif action == 2
        next_head_loc_m = head_loc_m;
        next_head_loc_n = head_loc_n - 1;
        next_move_dir   = 4;
    else
        next_head_loc_m = head_loc_m - 1;
        next_head_loc_n = head_loc_n;
        next_move_dir   = 1;
    end
end
next_head_loc = [next_head_loc_m, next_head_loc_n];
end