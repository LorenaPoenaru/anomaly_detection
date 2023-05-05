# #!/usr/bin/env Rscript

glr <- function() {
    print(1)

    # import the cpm package
    library('cpm')
    data <- read.csv("entropies.csv")

    x = data['entropies']
    # use the function to calculate the GLR statistics
    res = processStream(x, cpmType = 'GLR', ARL0 = 500, startup = 1500)
    # plot the photo and print the change points
    # plot(x, type='l')
    # abline(v = res$detectionTimes, lty=2)
    # abline(v = res$changePoints, lty=2, col='red')
    return(res$changePoints)
}