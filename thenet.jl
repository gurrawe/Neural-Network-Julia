using Random
using Plots

struct Parameters
  Name::String
  TestRate:: Float64
  Filename:: String
  L::Int64
  nr::Vector{Int64} 
  epochs::Int64
  n::Float64
  α::Float64
  batch_size::Int64
end 

mutable struct NeuralNet
    L::Int64                        # number of layers
    n::Vector{Int64}                # sizes of layers
    h::Vector{Vector{Float64}}      # units field
    ξ::Vector{Vector{Float64}} 
    b_ξ::Vector{Array{Float64,2}}     # units batched activation
    w::Vector{Array{Float64,2}}     # weights
    θ::Vector{Vector{Float64}}      # thresholds
    d_w::Vector{Array{Float64,2}}   #change of weights
    d_θ::Vector{Vector{Float64}}    #change of thresholds
    Δ::Vector{Vector{Float64}}
    b_Δ::Vector{Array{Float64,2}}    #propagation of errors
    d_w_prev::Vector{Array{Float64,2}} #prev change of weights
    d_θ_prev::Vector{Vector{Float64}}
    training_data::Vector{Vector{Float64}}
    test_data::Vector{Vector{Float64}} #prev change of thresholds
    min::Float64
    max::Float64
  end

function Parameters()
  nr = []
  S = ""
  Name = ""
  TestRate = 0.0
  Filename = ".."
  batch_size = 1
  epochs = 1
  n = 0.0
  α=0.0
  L = 1

  open(ARGS[1]) do f
 
    # line_number
    line = 1
    
    while ! eof(f)         
       S = readline(f)
       if(line ==1)
       Name = S
       elseif(line == 2)
       TestRate = parse(Float64,S)
      elseif(line == 3)
        Filename = S
      elseif(line == 4)
        batch_size = parse(Int64,S)
      elseif(line == 5)
        epochs = parse(Int64,S)
      elseif(line == 6)
        n = parse(Float64,S)
      elseif(line == 7)
        α= parse(Float64,S)
      elseif(line == 8)
        L = parse(Int64,S)
      elseif(line > 8)
         append!(nr,parse(Int64,S))
       end
       line = line +1
    end
  end
    return Parameters(Name, TestRate, Filename, L, nr, epochs, n, α, batch_size)
end 

  function scaling(xmin::Float64, xmax::Float64, x::Float64)::Float64 #eq. 1
    return 0.1 +  0.8*(x-xmin)/(xmax-xmin)
  end
  
  function descaling(xmin::Float64, xmax::Float64, s::Float64)::Float64
    return xmin + (xmax-xmin)/0.8*(s-0.1)
  end
      
  function NeuralNet(layers::Vector{Int64}, file::String, batch_size::Int64, data_ratio::Float64)
    L = length(layers)
    n = copy(layers)
  
    h = Vector{Float64}[]
    ξ = Vector{Float64}[]
    θ = Vector{Float64}[]
    d_θ = Vector{Float64}[]
    d_θ_prev = Vector{Float64}[]
    Δ = Vector{Float64}[]
    b_Δ = Array{Float64,2}[]
    b_ξ = Array{Float64,2}[]
    training_data = Vector{Float64}[]
    test_data = Vector{Float64}[]


    for ℓ in 1:L
      push!(h, zeros(layers[ℓ]))
      push!(ξ, zeros(layers[ℓ]))
      push!(θ, rand(layers[ℓ]))
      push!(d_θ, zeros(layers[ℓ]))
      push!(d_θ_prev, zeros(layers[ℓ]))
      push!(Δ, zeros(layers[ℓ]))
                         
    end
  
    w = Array{Float64,2}[]
    d_w = Array{Float64,2}[]
    d_w_prev = Array{Float64,2}[]

    push!(w, zeros(1, 1))
    push!(d_w, zeros(1, 1))
    push!(d_w_prev, zeros(1, 1))

    for ℓ in 2:L
      push!(w, rand(layers[ℓ], layers[ℓ - 1]))    
      push!(d_w, zeros(layers[ℓ], layers[ℓ - 1]))
      push!(d_w_prev, zeros(layers[ℓ], layers[ℓ - 1])) 
    end

    for p in 1:L
      push!(b_Δ, zeros(batch_size, layers[p]))
      push!(b_ξ, zeros(batch_size, layers[p]))
    end

    open(file) do f
 
      # line_number
      line = 0  
      TL = []
      while ! eof(f)         
         s = readline(f)
         if(line != 0)
         v = split(s)
        T = []
        for i in 1:length(v)
          append!(T,parse(Float64, v[i]))
        end 
        push!(training_data, T)
         end
         line = line +1
      end


      start = length(training_data)-convert(Int64,floor(length(training_data)*data_ratio))

      test_data = training_data[start : length(training_data)-1]
      training_data = training_data[1:start-1]
      
    end
    min = 0.0
    max = 0.0
  
    return NeuralNet(L, n, h, ξ, b_ξ, w, θ, d_w, d_θ, Δ, b_Δ, d_w_prev, d_θ_prev, training_data, test_data, min, max)
  end
  
  function convertData(nn::NeuralNet) #converting the data to numbers between 0-1. Also checks min and max value for the scaling.
      
      min = Vector{Float64}
      max = Vector{Float64}
      lo = length(nn.training_data)
      lp = length(nn.training_data[1])
      min = 999999
      max = 0

     for y in 1:lp
      for u in 1:length(nn.test_data)
        if(nn.test_data[u][y]>max)
          max =nn.test_data[u][y]

          end 
        if(nn.test_data[u][y]<min)
          min = nn.test_data[u][y]
          
        end
      end
      
        for u in 1:lo
          if(nn.training_data[u][y]>max)
            max =nn.training_data[u][y]
            
          end 

          if(nn.training_data[u][y]<min)
            min = nn.training_data[u][y]
             
          end
        end
        
      for u in 1:lo
        nn.training_data[u][y] = scaling(convert(Float64,min),convert(Float64,max),nn.training_data[u][y]) 
      end
      for u in 1:length(nn.test_data)
        nn.test_data[u][y] = scaling(convert(Float64,min),convert(Float64,max),nn.test_data[u][y]) 
      end

      nn.min = min
      nn.max = max

      min = 9999999
      max = 0

      
    end
   
    end
  
     
    
  

  function sigmoid(h::Float64)::Float64
    return 1 / (1 + exp(-h))
  end
  
  
  function feed_forward!(nn::NeuralNet, x_in::Vector{Float64}, y_out::Vector{Float64}, batch_it::Int64)
    # copy input to first layer, Eq. (6)
    nn.ξ[1] .= x_in
    
    for k in 1:length(x_in)
      nn.b_ξ[1][batch_it, k] = x_in[k] 
    end
  
    # feed-forward of input pattern
    for ℓ in 2:nn.L
      for i in 1:nn.n[ℓ]
        # calculate input field to unit i in layer ℓ, Eq. (8)
        h = -nn.θ[ℓ][i]
        for j in 1:nn.n[ℓ - 1]
          h += nn.w[ℓ][i, j] * nn.ξ[ℓ - 1][j]
        end
        # save field and calculate activation, Eq. (7)
        nn.h[ℓ][i] = h
        nn.ξ[ℓ][i] = sigmoid(h)
        nn.b_ξ[ℓ][batch_it,i] = nn.ξ[ℓ][i]
      end
    end
  
    # copy activation in output layer as output, Eq. (9)
    y_out .= nn.ξ[nn.L]
  end

  function gprim(h::Float64)::Float64
    return sigmoid(h)*(1-sigmoid(h)) #Eq. 13
  end


  #Batched backward propagation also used for single back propogation. 
  function batched_bp(nn::NeuralNet, z::Vector{Vector{Float64}}, batch_size::Int64, n::Float64, α::Float64)

    count = 0;
    for d in 1:batch_size
    for i in 1:nn.n[nn.L]
        nn.b_Δ[nn.L][d,i] = gprim(nn.h[nn.L][i])*(nn.b_ξ[nn.L][d,i]-z[d][i]) #Eq. 11   
    end
  end

    for ℓ in nn.L-1:-1:1
      for d in 1:batch_size
        for j in 1:nn.n[ℓ] 
            for i in 1:nn.n[ℓ+1]
                count = count + (nn.b_Δ[ℓ+1][d,i]*nn.w[ℓ+1][i,j]) #Eq. 12 
            end
                nn.b_Δ[ℓ][d,j] = gprim(nn.h[ℓ][j])*count #Eq. 12
                count = 0
        end
      end 
    end

    for ℓ in nn.L-1:-1:1

      for j in 1:nn.n[ℓ] 
          for i in 1:nn.n[ℓ+1]
            t_w = 0
            for d in 1:batch_size
              t_w += nn.b_Δ[ℓ+1][d,i]*nn.b_ξ[ℓ][d,j]
            end
              
              nn.d_w[ℓ+1][i,j] = -n*t_w+α*nn.d_w_prev[ℓ+1][i,j] #14
              nn.d_w_prev[ℓ+1][i,j] = nn.d_w[ℓ+1][i,j]
              nn.w[ℓ+1][i,j] = nn.w[ℓ+1][i,j] + nn.d_w[ℓ+1][i,j]  

          end
      end

      for i in 1:nn.n[ℓ+1]
        t_θ = 0
        for d in 1:batch_size
          t_θ += nn.b_Δ[ℓ+1][d,i]
        end 
          nn.d_θ[ℓ+1][i] = n*t_θ+α*nn.d_θ_prev[ℓ+1][i]
          nn.d_θ_prev[ℓ+1][i] = nn.d_θ[ℓ+1][i]
          nn.θ[ℓ+1][i] = nn.θ[ℓ+1][i] + nn.d_θ[ℓ+1][i]
      end
  end

  end

  function Printθ(nn::NeuralNet)
    open("thresholds.txt", "w") do iop
      
  
          for ℓ in nn.L-1:-1:1 
            for i in 1:nn.n[ℓ+1]
              temp1 = ℓ+1

              temp3 = i

              temp5 = nn.θ[ℓ+1][i]
              s = "Threshould at layer $temp1, at node $temp3. The value is: $temp5 \n"
              write(iop,s)
            end

          end
    end;
  end

function PrintW(nn::NeuralNet)
  open("weights.txt", "w") do iop
    

  for ℓ in nn.L-1:-1:1
      for j in 1:nn.n[ℓ] 
          for i in 1:nn.n[ℓ+1]
            temp1 = ℓ+1
            temp2 = ℓ
            temp3 = i
            temp4 = j
            temp5 = nn.w[ℓ+1][i,j]
            s = "Weight between layer $temp1 and $temp2, from node $temp3 to $temp4. The value is: $temp5 \n"
            write(iop,s)
          end
      end
    end
    end;
  end
  
  
function Mainen(nt,αt,t)::Float64
  p = Parameters()
  Error = 0.0
  y_list = []
  real_values = []
  plotlist = []
  println("Running ", p.Name)
  println("With data from ", p.Filename)
  println("With layers", p.nr)
  println("With learning rate n ", p.n)
  println("With lmomentum α ", p.α)
  println("With batch_size ", p.batch_size)
  println("With epochs ", p.epochs)
  println("With ",100*p.TestRate,"% for testing rest for training")

  if(t == 0)
  n = p.n
  α = p.α
else
  println("This is inside a test run for parameters")
  n = nt
  α = αt
  println(n)
  println(α)
end 

    batch_size = p.batch_size
    layers = p.nr
    nn = NeuralNet(layers, p.Filename, batch_size, p.TestRate)
    convertData(nn)
    sizen = length(nn.training_data[1])
    y_out = zeros(nn.n[nn.L])

    for m in 1:p.epochs
      if(m % 100 == 0)
        batch_size = convert(Int64,floor(batch_size/2))
        if(batch_size < 1)
          batch_size = 1
        end
      end
      nn.training_data .= shuffle(nn.training_data)

      for v in 1:batch_size:length(nn.training_data)-batch_size
        
        the_whole = Vector{Float64}[]
        for d in 1:batch_size
          feed_forward!(nn,nn.training_data[v+d-1][1:sizen-1] , y_out, d)
          v1 = [nn.training_data[v+d-1][sizen]]
          append!(the_whole, [v1]) 
        end

        batched_bp(nn, the_whole, batch_size, n, α)

      end
     

    Error = 0.0
    ErrorT = 0.0
    ErrorN = 0.0
    

    for v in 1:length(nn.test_data) #385
      feed_forward!(nn,nn.test_data[v][1:sizen-1], y_out, 1)
      ErrorT += abs(y_out[1] - nn.test_data[v][sizen])
      ErrorN += nn.test_data[v][sizen]

      if(m ==  p.epochs) #
        append!(y_list, y_out[1])
        append!(real_values, nn.test_data[v][sizen])
      end
    end
    Error = 100 *(ErrorT/ErrorN)
    append!(plotlist, Error)

  end 
  #PrintW(nn)
  #Printθ(nn)
  println("The error rate is ", plotlist[length(plotlist)])

  plot(1:p.epochs, plotlist)
  savefig("LearningCurveNN.png")
  #plot scatter plot
  scatter(real_values, y_list)

  xlabel!("True Values")
  ylabel!("Predicted Values")

  title!("Scatter plot")
  savefig("scatterNN.png")







    s = "Error-rate $Error"
    open("resultsNN.txt", "w") do io
      for j in 1:length(y_list)
        temp1 = y_list[j]
        temp2 = nn.test_data[j][sizen]
        s1="Predicted value: $temp1  Real value: $temp2\n"
        write(io, s1)
      end 
      write(io, s)
    end;
    return plotlist[length(plotlist)]
end


function Test()
best_n = 0.01
best_α = 0.0
best_error = 9999.0
error = 999.9

n = 0.01
α = 0.1
while n < 0.21
  while α < 0.91
    error = Mainen(n, α, 1)
    if error < best_error
      best_error = error
      best_n = n
      best_α = α
    end
    println("Best run with: ")
    println(best_n)
    println(best_α)
    α += 0.1
    end
    α = 0.1
    n += 0.03
  end
  println("Best rate: ", best_error, " n: ", best_n, " α: ", best_α)
end 


#Test() #Run to calculate best params

plotlist = Mainen(0,0,0) #Run to go with learning rate and momentum from parameters file






  